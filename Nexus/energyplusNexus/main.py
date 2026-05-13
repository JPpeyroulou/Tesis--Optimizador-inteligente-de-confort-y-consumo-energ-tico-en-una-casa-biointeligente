"""
EnergyPlus–Home Assistant Nexus.

Flujo:
1. Simula un anio entero con EnergyPlus (una vez).
2. Por cada hora (paso): lee actuadores de HA -> sube los sensores de esa hora a HA.
3. Repite desde 1 en bucle hasta Ctrl+C.  --once = un solo anio y salir.

Uso:
  python main.py
  python main.py --once
  python main.py --verbose   # log detallado por entidad en cada paso
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from ha_client import (
    ACTUATORS,
    SENSORS,
    create_ha_session,
    default_actuators,
    fetch_actuators,
    set_forecast_24h_csv,
    push_sensors,
    set_simulation_datetime,
    set_sync_flag_on,
    wait_for_sync_flag_off,
)
from parse_output import extract_sensor_values_all_hours, parse_datetime_string
from run_energyplus import run as run_energyplus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config no encontrado: {p}")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="EnergyPlus–HA Nexus")
    ap.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta a config.yaml (default: config.yaml)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo verificar config, HA y archivos; no ejecutar EP ni subir a HA.",
    )
    ap.add_argument(
        "--no-push",
        action="store_true",
        help="En cada paso solo leer actuadores; no subir sensores a HA.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Un solo anio (simular + todos los pasos) y salir.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Log detallado: cada lectura y escritura por entidad en cada paso.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=0,
        metavar="N",
        help="Maximo de pasos (horas) por anio; 0 = todos. Util para pruebas.",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="No usar bandera de sincronizacion (avanzar sin esperar al modelo AI).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ha = cfg.get("home_assistant") or {}
    ep = cfg.get("energyplus") or {}

    ha_url = ha.get("url", "http://localhost:8123")
    ha_token = ha.get("token", "")
    sync_entity = "" if args.no_sync else (ha.get("sync_flag_entity") or "").strip()
    use_input_number = ha.get("use_input_number_for_sensors", True)
    use_websocket = ha.get("use_websocket", True)
    exe = ep.get("executable", "EnergyPlus")
    idf_path = Path(ep.get("idf_path", "idf_multiplesZonas.epJSON"))
    weather_path = Path(ep.get("weather_path", ""))
    run_dir = Path(ep.get("run_directory", "eplus_run"))
    simulation_year = int(ep.get("simulation_year", 2021))
    start_month = int(ep.get("start_month", 1))
    start_day = int(ep.get("start_day", 1))

    if not ha_token:
        logger.warning("home_assistant.token vacio; las llamadas a HA pueden fallar.")
    if not args.dry_run and (not weather_path or weather_path.suffix.lower() != ".epw"):
        raise SystemExit(
            "Configura energyplus.weather_path con un archivo .epw en config.yaml"
        )

    base = Path(__file__).resolve().parent
    if not idf_path.is_absolute():
        idf_path = base / idf_path
    if weather_path and str(weather_path).strip():
        weather_path = Path(weather_path)
        if not weather_path.is_absolute():
            weather_path = base / weather_path
    if not run_dir.is_absolute():
        run_dir = base / run_dir

    # --- Dry-run ---
    if args.dry_run:
        logger.info("=== DRY RUN (solo verificacion) ===")
        logger.info("Config cargado; comprobando archivos y HA...")
        ok = True
        if not idf_path.is_file():
            logger.error("IDF no encontrado: %s", idf_path)
            ok = False
        else:
            logger.info("IDF OK: %s", idf_path)
        if weather_path and not weather_path.is_file():
            logger.warning("Weather no encontrado: %s", weather_path)
        elif weather_path:
            logger.info("Weather OK: %s", weather_path)
        exe_path = Path(exe)
        if not exe_path.is_file():
            logger.warning("EnergyPlus no encontrado: %s", exe)
        else:
            logger.info("EnergyPlus OK: %s", exe)
        try:
            sess = create_ha_session(ha_url, ha_token, use_websocket)
            try:
                acts = fetch_actuators(ha_url, ha_token, verbose=True, session=sess)
            finally:
                sess.close()
            logger.info("HA OK. Resumen actuadores: %s", acts)
        except Exception as e:
            logger.error("HA fallo: %s", e)
            ok = False
        if ok:
            logger.info("Dry-run OK. Ejecuta sin --dry-run.")
        raise SystemExit(0 if ok else 1)

    verbose = args.verbose
    run_loop = not args.once
    if run_loop:
        logger.info("Modo continuo: anio tras anio. Ctrl+C para detener.")
    else:
        logger.info("Modo --once: un solo anio y salir.")
    if sync_entity:
        logger.info("Sincronizacion con bandera: %s (espera indefinida)", sync_entity)

    stop = False
    ha_session = None

    def on_sig(sig: int, frame: object) -> None:
        nonlocal stop
        stop = True
        logger.info("")
        logger.info("Ctrl+C recibido. Cerrando tras este paso...")

    if run_loop:
        signal.signal(signal.SIGINT, on_sig)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, on_sig)

    try:
        ha_session = create_ha_session(ha_url, ha_token, use_websocket)
        year_num = 1
        while True:
            logger.info("")
            logger.info("========== ANIO %d: Simular EnergyPlus (ano entero) ==========", year_num)
            run_energyplus(
                executable=exe,
                idf_path=idf_path,
                weather_path=weather_path,
                run_directory=run_dir,
                readvars=True,
            )
            all_hours = extract_sensor_values_all_hours(run_dir)
            # Saltar "design days" / warmup inicial hasta fecha de inicio configurada.
            start_idx = None
            for i, (dt_str, _s) in enumerate(all_hours):
                parsed_dt = parse_datetime_string(dt_str)
                if parsed_dt:
                    month, day, _hour, _minute = parsed_dt
                    if month == start_month and day == start_day:
                        start_idx = i
                        break
            if start_idx is not None and start_idx > 0:
                logger.info(
                    "  -> Saltando %d pasos iniciales (design days) hasta %02d/%02d.",
                    start_idx,
                    start_day,
                    start_month,
                )
                all_hours = all_hours[start_idx:]
            if args.max_steps > 0:
                all_hours = all_hours[: args.max_steps]
                logger.info("  -> %d pasos (horas) por --max-steps.", len(all_hours))
            else:
                logger.info("  -> %d pasos (horas) listos.", len(all_hours))
            logger.info("")

            step_log_path = run_dir / "sensors_output_by_step.csv"
            with step_log_path.open("w", newline="", encoding="utf-8") as step_log_f:
                step_writer = csv.writer(step_log_f)
                step_writer.writerow(["step", "simulation_datetime"] + list(SENSORS) + list(ACTUATORS))
            logger.info("Log sensores/actuadores por paso: %s", step_log_path)
            forecast_log_path = run_dir / "forecast_by_step.csv"
            with forecast_log_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["step", "simulation_datetime", "forecast_offset", "forecast_datetime", "outdoor_temp_C", "outdoor_humidity_pct"]
                )
            logger.info("Log forecast (dia/hora) por paso: %s", forecast_log_path)

            for step, (dt_str, sensors) in enumerate(all_hours):
                if stop:
                    break
                n = len(all_hours)
                if verbose:
                    logger.info("")
                    logger.info("=== PASO %d/%d (%s) ===", step + 1, n, dt_str)
                if sync_entity and step > 0:
                    logger.info("  Esperando bandera OFF (AI listo)...")
                    wait_for_sync_flag_off(ha_url, ha_token, sync_entity, session=ha_session)
                if step == 0:
                    acts = default_actuators()
                    if verbose:
                        logger.info("  Actuadores (primer paso, en 0): %s", acts)
                else:
                    if verbose:
                        logger.info("  Leer actuadores de HA:")
                    acts = fetch_actuators(ha_url, ha_token, verbose=verbose, session=ha_session)
                if verbose:
                    logger.info("  -> Resumen: %s", acts)
                    if not args.no_push:
                        logger.info("  Escribir sensores en HA:")
                parsed = parse_datetime_string(dt_str)
                sim_datetime = ""
                if parsed:
                    month, day, hour, minute = parsed
                    sim_datetime = f"{simulation_year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00"
                else:
                    sim_datetime = dt_str.strip()
                with step_log_path.open("a", newline="", encoding="utf-8") as step_log_f:
                    csv.writer(step_log_f).writerow(
                        [step + 1, sim_datetime]
                        + [sensors.get(s, "") for s in SENSORS]
                        + [acts.get(a, "") for a in ACTUATORS]
                    )
                forecast_rows: list[tuple[float, float, float, float, float, float]] = []
                for k in range(24):
                    idx = min(step + k, len(all_hours) - 1)
                    s = all_hours[idx][1]
                    temp_air = s.get("sensor.outdoor_temperature_sensor", 0.0)
                    rh = s.get("sensor.outdoor_humidity_sensor", 0.0)
                    forecast_rows.append((temp_air, rh, 0.0, 0.0, 0.0, 0.0))
                if parsed:
                    month, day, hour, minute = parsed
                    # EnergyPlus puede reportar 24:00:00; normalizar a 00:00 del dia siguiente.
                    day_shift = 0
                    if hour >= 24:
                        hour = 0
                        day_shift = 1
                    base_dt = datetime(simulation_year, month, day, hour, minute or 0) + timedelta(days=day_shift)
                    with forecast_log_path.open("a", newline="", encoding="utf-8") as fl:
                        w = csv.writer(fl)
                        for k in range(min(24, len(forecast_rows))):
                            fd = base_dt + timedelta(hours=k)
                            r = forecast_rows[k]
                            w.writerow([step + 1, sim_datetime, k, fd.strftime("%Y-%m-%d %H:%M"), round(r[0], 4), round(r[1], 4)])
                if args.no_push:
                    if not verbose:
                        logger.info("PASO %d/%d (%s): actuadores en 0 (no-push, sin escribir)", step + 1, n, dt_str)
                else:
                    push_sensors(
                        ha_url, ha_token, sensors,
                        verbose=verbose,
                        use_input_number=use_input_number,
                        session=ha_session,
                    )
                    if parsed:
                        month, day, hour, minute = parsed
                        if hour >= 24:
                            hour = 0
                            dt_norm = datetime(simulation_year, month, day, hour, minute or 0) + timedelta(days=1)
                            month, day = dt_norm.month, dt_norm.day
                        set_simulation_datetime(
                            ha_url, ha_token,
                            simulation_year, month, day, hour, minute,
                            session=ha_session,
                        )
                    set_forecast_24h_csv(ha_url, ha_token, forecast_rows, session=ha_session)
                    if sync_entity:
                        set_sync_flag_on(ha_url, ha_token, sync_entity, session=ha_session)
                    if not verbose:
                        if step == 0:
                            logger.info("PASO %d/%d (%s): actuadores en 0, escritos %d sensores", step + 1, n, dt_str, len(sensors))
                        else:
                            logger.info(
                                "PASO %d/%d (%s): leidos 5 actuadores, escritos %d sensores",
                                step + 1, n, dt_str, len(sensors),
                            )

            if stop or not run_loop:
                break
            year_num += 1

        logger.info("")
        logger.info("Listo.")
    finally:
        if ha_session is not None:
            ha_session.close()


if __name__ == "__main__":
    main()
