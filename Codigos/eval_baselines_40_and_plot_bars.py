#!/usr/bin/env python3
"""Evalua solo baselines_setpoints_40 y genera barras de costo/PMV."""

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = "/workspaces/sinergym"
YEAR_FOR_WEEKDAY = 2000
PLOT_YEAR = 2025
PMV_OUTER_BAND = 1.5
ZONE_STYLE = {
    "east": {"label": "Este", "color": "#f4a261"},
    "west": {"label": "Oeste", "color": "#2ca02c"},
    "north": {"label": "Norte", "color": "#4e79a7"},
    "south": {"label": "Sur", "color": "#e15759"},
}


@dataclass
class ResultPoint:
    label: str
    source: str
    samples: int
    costo_anual: float
    pmv_dev_acum: float
    status: str
    message: str


def load_tarifa(path: str):
    with open(path, "r", encoding="utf-8") as f:
        t = json.load(f)

    dias_map = {
        "lunes": 0,
        "martes": 1,
        "miercoles": 2,
        "jueves": 3,
        "viernes": 4,
        "sabado": 5,
        "domingo": 6,
    }
    precio_punta = float(t["precios"]["punta"])
    precio_fuera = float(t["precios"]["fuera_de_punta"])
    punta_ini = int(t["horarios"]["punta_inicio"])
    punta_fin = int(t["horarios"]["punta_fin"])
    dias_punta = {dias_map[d] for d in t["horarios"]["dias_punta"]}
    return precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta


def price_at(
    month: int,
    day: int,
    hour: int,
    precio_punta: float,
    precio_fuera: float,
    punta_ini: int,
    punta_fin: int,
    dias_punta: set[int],
) -> float:
    dt = datetime(YEAR_FOR_WEEKDAY, month, max(1, min(28, day)), hour)
    if dt.weekday() in dias_punta and punta_ini <= hour <= punta_fin:
        return precio_punta
    return precio_fuera


def pmv_simple(tdb: float, rh: float) -> float:
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh


def pmv_deviation(pmv: float) -> float:
    return max(-0.5 - pmv, 0.0) + max(pmv - 0.5, 0.0)


def _pick_first_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"No se encontro columna esperada: {candidates}")


def detect_active_zones_from_epjson(epjson_path: str) -> list[str]:
    """Return active perimeter zones based on On/Off schedules in epJSON."""
    with open(epjson_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    schedules = model.get("Schedule:Compact", {})
    schedule_to_zone = {
        "ZonaEast": "east",
        "ZonaWest": "west",
        "ZonaNorth": "north",
        "ZonaSouth": "south",
    }

    active_zones = []
    for schedule_name, zone_name in schedule_to_zone.items():
        obj = schedules.get(schedule_name)
        if not obj:
            continue
        data = obj.get("data", [])
        numeric_values = [item.get("field") for item in data if isinstance(item.get("field"), (int, float))]
        if any(float(v) > 0.0 for v in numeric_values):
            active_zones.append(zone_name)

    return active_zones


def compute_metrics_from_eplusout(eplusout_csv: str, tarifa_json: str, active_zones: list[str]):
    precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta = load_tarifa(tarifa_json)

    pmv_dev_acum = 0.0
    costo_acum = 0.0
    samples = 0

    if not active_zones:
        raise ValueError("No hay zonas activas para evaluar PMV.")

    with open(eplusout_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        power_col = _pick_first_column(
            cols,
            [
                "BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)",
                "Heat Pump Electricity Rate [W](Hourly)",
            ],
        )
        zone_columns = {}
        zone_to_candidates = {
            "east": (
                ["EAST PERIMETER:Zone Air Temperature [C](Hourly)"],
                ["EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
            ),
            "west": (
                ["WEST PERIMETER:Zone Air Temperature [C](Hourly)"],
                ["WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
            ),
            "north": (
                ["NORTH PERIMETER:Zone Air Temperature [C](Hourly)"],
                ["NORTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
            ),
            "south": (
                ["SOUTH PERIMETER:Zone Air Temperature [C](Hourly)"],
                ["SOUTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
            ),
        }
        for zone in active_zones:
            t_candidates, h_candidates = zone_to_candidates[zone]
            zone_columns[zone] = (
                _pick_first_column(cols, t_candidates),
                _pick_first_column(cols, h_candidates),
            )

        for r in reader:
            dt = r["Date/Time"].strip().split()
            month, day = dt[0].split("/")
            hour = int(dt[1].split(":")[0]) - 1
            if hour < 0:
                hour = 0

            power_w = float(r[power_col])
            step_pmv_dev = 0.0
            for zone in active_zones:
                t_col, h_col = zone_columns[zone]
                pmv = pmv_simple(float(r[t_col]), float(r[h_col]))
                step_pmv_dev += pmv_deviation(pmv)
            pmv_dev_acum += step_pmv_dev

            pr = price_at(
                int(month),
                int(day),
                hour,
                precio_punta,
                precio_fuera,
                punta_ini,
                punta_fin,
                dias_punta,
            )
            costo_acum += (power_w / 1000.0) * pr
            samples += 1

    return pmv_dev_acum, costo_acum, samples


def _zone_column_candidates():
    return {
        "east": (
            ["EAST PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "west": (
            ["WEST PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "north": (
            ["NORTH PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["NORTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "south": (
            ["SOUTH PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["SOUTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
    }


def _parse_eplus_datetime(dt_raw: str) -> datetime:
    parts = dt_raw.strip().split()
    month, day = parts[0].split("/")
    hour = int(parts[1].split(":")[0]) - 1
    if hour < 0:
        hour = 0
    return datetime(PLOT_YEAR, int(month), int(day), hour)


def _parse_setpoint_from_label(label: str) -> float | None:
    m = re.search(r"sp(\d+)", label)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def save_pmv_indoor_may_oct_plot(
    eplusout_csv: str, label: str, active_zones: list[str], out_dir: str
) -> str:
    """Grafica PMV interior por zona entre 1 mayo y 1 octubre (sin clima exterior)."""
    if not active_zones:
        raise ValueError("No hay zonas activas para graficar PMV interior.")

    zone_candidates = _zone_column_candidates()
    start_dt = datetime(PLOT_YEAR, 5, 1, 0, 0, 0)
    end_dt = datetime(PLOT_YEAR, 10, 1, 0, 0, 0)
    setpoint_temp = _parse_setpoint_from_label(label)

    # Mantiene solo la ultima aparicion por timestamp (los dias de diseño suelen
    # aparecer primero en el CSV y repetir fechas/horas del run anual).
    pmv_by_dt = {}
    rh_by_dt = {}

    with open(eplusout_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        zone_columns = {}
        for zone in active_zones:
            t_candidates, h_candidates = zone_candidates[zone]
            zone_columns[zone] = (
                _pick_first_column(cols, t_candidates),
                _pick_first_column(cols, h_candidates),
            )

        for r in reader:
            dt = _parse_eplus_datetime(r["Date/Time"])
            if not (start_dt <= dt < end_dt):
                continue
            zone_values = {}
            zone_rh_values = {}
            for zone in active_zones:
                t_col, h_col = zone_columns[zone]
                rh_val = float(r[h_col])
                pmv_val = pmv_simple(float(r[t_col]), rh_val)
                zone_values[zone] = pmv_val
                zone_rh_values[zone] = rh_val
            pmv_by_dt[dt] = zone_values
            rh_by_dt[dt] = zone_rh_values

    if not pmv_by_dt:
        raise ValueError(f"No hay datos entre 1 mayo y 1 octubre en {eplusout_csv}")

    datetimes = sorted(pmv_by_dt.keys())
    pmv_by_zone = {z: [pmv_by_dt[dt][z] for dt in datetimes] for z in active_zones}
    pmv_setpoint_by_zone = {}
    if setpoint_temp is not None:
        for z in active_zones:
            pmv_setpoint_by_zone[z] = [pmv_simple(setpoint_temp, rh_by_dt[dt][z]) for dt in datetimes]

    fig, axes = plt.subplots(len(active_zones), 1, figsize=(18, 4.2 * len(active_zones)), sharex=True)
    if len(active_zones) == 1:
        axes = [axes]

    for ax, zone in zip(axes, active_zones):
        style = ZONE_STYLE.get(zone, {"label": zone, "color": "#333333"})
        ax.plot(
            datetimes,
            pmv_by_zone[zone],
            color=style["color"],
            linewidth=1.8,
            alpha=0.95,
            label=f"PMV interior zona {style['label']}",
        )
        if zone in pmv_setpoint_by_zone:
            ax.plot(
                datetimes,
                pmv_setpoint_by_zone[zone],
                color="#111111",
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                label=f"PMV calculado con setpoint {setpoint_temp:.0f}C",
            )
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(-0.5, color="blue", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_ylabel("PMV")
        ax.set_title(f"Zona {style['label']}", loc="left", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(start_dt, end_dt)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("Mes")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))
    fig.suptitle(f"{label} - PMV interior (1 mayo a 1 octubre, sin clima exterior)", fontsize=14, fontweight="bold")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}_pmv_indoor_may01_oct01.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_pmv_category_percentages_from_eplusout(
    eplusout_csv: str, active_zones: list[str]
) -> dict[str, float]:
    """Calcula porcentajes PMV para 4 categorias desde eplusout.csv."""
    if not active_zones:
        raise ValueError("No hay zonas activas para calcular porcentajes PMV.")

    zone_to_candidates = {
        "east": (
            ["EAST PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "west": (
            ["WEST PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "north": (
            ["NORTH PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["NORTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
        "south": (
            ["SOUTH PERIMETER:Zone Air Temperature [C](Hourly)"],
            ["SOUTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        ),
    }

    count_total = 0
    count_comfort = 0
    count_minus1_1 = 0
    count_minus_outer = 0

    with open(eplusout_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        zone_columns = {}
        for zone in active_zones:
            t_candidates, h_candidates = zone_to_candidates[zone]
            zone_columns[zone] = (
                _pick_first_column(cols, t_candidates),
                _pick_first_column(cols, h_candidates),
            )

        for r in reader:
            for zone in active_zones:
                t_col, h_col = zone_columns[zone]
                pmv = pmv_simple(float(r[t_col]), float(r[h_col]))
                count_total += 1
                if -0.5 <= pmv <= 0.5:
                    count_comfort += 1
                if -1.0 <= pmv <= 1.0:
                    count_minus1_1 += 1
                if -PMV_OUTER_BAND <= pmv <= PMV_OUTER_BAND:
                    count_minus_outer += 1

    if count_total == 0:
        raise ValueError(f"No hay muestras PMV en {eplusout_csv}")

    count_between = max(0, count_minus1_1 - count_comfort)
    count_between_outer = max(0, count_minus_outer - count_minus1_1)
    count_rest = max(0, count_total - count_minus_outer)
    return {
        "comfort_pct": 100.0 * count_comfort / count_total,
        "between_minus1_1_pct": 100.0 * count_between / count_total,
        "between_minus_outer_pct": 100.0 * count_between_outer / count_total,
        "rest_pct": 100.0 * count_rest / count_total,
    }


def parse_setpoint_from_path(path: str):
    base = os.path.basename(os.path.dirname(path))  # sp_20
    if base.startswith("sp_"):
        try:
            return int(base.split("_")[1])
        except (ValueError, IndexError):
            return 10**9
    return 10**9


def discover_baselines(baseline_root: str):
    pattern = os.path.join(baseline_root, "sp_*", "eplusout.csv")
    paths = glob.glob(pattern)
    paths = sorted(paths, key=lambda p: (parse_setpoint_from_path(p), p))
    return paths


def save_global_bars(output_dir: str, points):
    ok_points = [p for p in points if p.status == "ok"]
    labels = [p.label for p in ok_points]
    costos = [p.costo_anual for p in ok_points]
    pmvs = [p.pmv_dev_acum for p in ok_points]

    fig_w = max(10, 0.8 * max(1, len(labels)))

    fig1, ax1 = plt.subplots(figsize=(fig_w, 6))
    ax1.bar(labels, costos, color="#1f77b4")
    ax1.set_ylabel("Costo anual acumulado ($)")
    ax1.set_title("Costo anual acumulado - baselines setpoints_40")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_cost = os.path.join(output_dir, "bar_costo_baselines_40.png")
    plt.savefig(out_cost, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(fig_w, 6))
    ax2.bar(labels, pmvs, color="#d62728")
    ax2.set_ylabel("Desviacion PMV acumulada anual")
    ax2.set_title("Desviacion PMV acumulada anual - baselines setpoints_40")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_pmv = os.path.join(output_dir, "bar_pmv_baselines_40.png")
    plt.savefig(out_pmv, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return out_cost, out_pmv


def save_comfort_stacked_bars_baselines(output_dir: str, points, active_zones: list[str]):
    """Genera barra apilada por baseline con 4 rangos de PMV."""
    ok_points = [p for p in points if p.status == "ok" and p.source and os.path.exists(p.source)]
    if not ok_points:
        raise ValueError("No hay baselines OK para generar barras de confort PMV.")

    labels = []
    comfort = []
    between = []
    between_outer = []
    rest = []
    for p in ok_points:
        pct = compute_pmv_category_percentages_from_eplusout(p.source, active_zones)
        labels.append(p.label)
        comfort.append(pct["comfort_pct"])
        between.append(pct["between_minus1_1_pct"])
        between_outer.append(pct["between_minus_outer_pct"])
        rest.append(pct["rest_pct"])

    x = np.arange(len(labels))
    b1 = np.array(comfort, dtype=float)
    b2 = np.array(between, dtype=float)
    b3 = np.array(between_outer, dtype=float)
    b4 = np.array(rest, dtype=float)

    fig_w = max(10, 0.8 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    width = 0.75
    ax.bar(x, b1, color="#2ca02c", width=width, label="Confort [-0.5, 0.5]")
    ax.bar(x, b2, bottom=b1, color="#ffbf00", width=width, label="Entre -1 y 1 (fuera confort)")
    ax.bar(
        x,
        b3,
        bottom=b1 + b2,
        color="#d62728",
        width=width,
        label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
    )
    ax.bar(x, b4, bottom=b1 + b2 + b3, color="#8e44ad", width=width, label="Resto")

    for i, (v1, v2, v3, v4) in enumerate(zip(b1, b2, b3, b4)):
        if v1 >= 4:
            ax.text(
                i, v1 / 2.0, f"{v1:.1f}%", ha="center", va="center",
                fontsize=9, rotation=90, color="white", fontweight="bold"
            )
        if v2 >= 4:
            ax.text(
                i, v1 + v2 / 2.0, f"{v2:.1f}%", ha="center", va="center",
                fontsize=9, rotation=90, color="black", fontweight="bold"
            )
        if v3 >= 4:
            ax.text(
                i, v1 + v2 + v3 / 2.0, f"{v3:.1f}%", ha="center", va="center",
                fontsize=9, rotation=90, color="white", fontweight="bold"
            )
        if v4 >= 4:
            ax.text(
                i, v1 + v2 + v3 + v4 / 2.0, f"{v4:.1f}%", ha="center", va="center",
                fontsize=9, rotation=90, color="white", fontweight="bold"
            )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Distribucion PMV por baseline (barras apiladas)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "bar_confort_pmv_baselines_40_stacked.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_summary_csv(output_dir: str, points):
    out_csv = os.path.join(output_dir, "resumen_baselines_40.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "source",
                "samples",
                "costo_anual_acumulado",
                "pmv_desviacion_acumulada_anual",
                "status",
                "message",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.label,
                    p.source,
                    p.samples,
                    p.costo_anual,
                    p.pmv_dev_acum,
                    p.status,
                    p.message,
                ]
            )
    return out_csv


def main():
    parser = argparse.ArgumentParser(
        description="Evalua solo baseline_setpoints_40 y genera barras globales."
    )
    parser.add_argument(
        "--baseline-root",
        default=f"{ROOT}/baseline_setpoints_ae140bxydeg_north-east-west",
        help="Directorio raiz con carpetas sp_XX/eplusout.csv.",
    )
    parser.add_argument(
        "--tarifa-json",
        default=f"{ROOT}/sinergym/data/tarifas/tarifas_ute.json",
        help="Path al JSON de tarifas.",
    )
    parser.add_argument(
        "--building-epjson",
        default=f"{ROOT}/sinergym/data/buildings/idf_multiplesZonas_termostato_ae140bxydeg.epJSON",
        help="Modelo epJSON para detectar zonas activas (ZonaEast/West/North/South).",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{ROOT}/baseline_setpoints_ae140bxydeg_north-east-west",
        help="Directorio de salida. Default: /workspaces/sinergym/baseline_setpoints_ae140bxydeg_north-east-west",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or f"{ROOT}/batch_eval_baselines_ae140bxydeg_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    active_zones = detect_active_zones_from_epjson(args.building_epjson)
    if not active_zones:
        raise ValueError(
            f"No se detectaron zonas activas en {args.building_epjson}. "
            "Revisa Schedule:Compact (ZonaEast/West/North/South)."
        )
    print(f"[INFO] Zonas activas detectadas: {', '.join(active_zones)}")

    points = []
    may_oct_plot_paths = []
    baseline_paths = discover_baselines(args.baseline_root)

    if not baseline_paths:
        raise FileNotFoundError(
            f"No se encontraron baselines en: {args.baseline_root}. "
            "Esperado: <baseline-root>/sp_XX/eplusout.csv"
        )

    for path in baseline_paths:
        sp = parse_setpoint_from_path(path)
        label = f"BL_sp{sp}" if sp != 10**9 else f"BL_{os.path.basename(os.path.dirname(path))}"
        try:
            pmv_dev, costo, samples = compute_metrics_from_eplusout(
                path, args.tarifa_json, active_zones
            )
            points.append(ResultPoint(label, path, samples, costo, pmv_dev, "ok", ""))
            print(f"[INFO] {label}: OK ({samples} muestras)")
            try:
                out_plot = save_pmv_indoor_may_oct_plot(
                    eplusout_csv=path,
                    label=label,
                    active_zones=active_zones,
                    out_dir=os.path.join(output_dir, "plots_pmv_may01_oct01_sin_exterior"),
                )
                may_oct_plot_paths.append(out_plot)
            except Exception as plot_err:
                print(f"[WARN] {label}: no se pudo generar PMV interior mayo-octubre: {plot_err}")
        except Exception as e:
            msg = f"Error baseline: {e}"
            print(f"[WARN] {label}: {msg}")
            points.append(ResultPoint(label, path, 0, 0.0, 0.0, "error", msg))

    out_csv = save_summary_csv(output_dir, points)
    out_cost, out_pmv = save_global_bars(output_dir, points)
    out_confort = save_comfort_stacked_bars_baselines(output_dir, points, active_zones)

    ok_n = len([p for p in points if p.status == "ok"])
    err_n = len([p for p in points if p.status == "error"])
    print("\n=== RESUMEN ===")
    print(f"Total baselines: {len(points)} | OK: {ok_n} | Error: {err_n}")
    print(f"CSV resumen: {out_csv}")
    print(f"Barra costo: {out_cost}")
    print(f"Barra PMV: {out_pmv}")
    print(f"Barra confort PMV: {out_confort}")
    if may_oct_plot_paths:
        print("Graficas PMV interior (1 mayo a 1 octubre, sin clima exterior):")
        for p in may_oct_plot_paths:
            print(f" - {p}")


if __name__ == "__main__":
    main()
