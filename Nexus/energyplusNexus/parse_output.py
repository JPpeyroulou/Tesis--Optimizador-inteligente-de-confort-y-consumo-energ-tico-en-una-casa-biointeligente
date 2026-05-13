"""
Parsea salidas de EnergyPlus (ESO o CSV) y devuelve un diccionario
sensor_entity_id -> valor para subir a Home Assistant.

Mapeo según idf_multiplesZonas.epJSON (Output:Variable, Output:Meter):
- Zone Radiant HVAC Heating Rate -> north/south/east/west_heating_rate_sensor (W; HA usa %)
- Site Outdoor Air Drybulb / Relative Humidity -> outdoor_temperature_sensor, outdoor_humidity_sensor
- Zone Air Temperature / Relative Humidity (NORTH/SOUTH/EAST/WEST PERIMETER) -> zona
- Heat Pump Electricity Rate (BOMBACALOR_HP) -> heat_pump_power_sensor, total_electricity_hvac_sensor
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# (variable_name, key) -> entity_id. Keys en CSV suelen ir en MAYÚSCULAS (ej. LOZARADIANTE_ZONANORTH).
EP_TO_HA: list[tuple[str, str, str]] = [
    ("Site Outdoor Air Drybulb Temperature", "ENVIRONMENT", "sensor.outdoor_temperature_sensor"),
    ("Site Outdoor Air Relative Humidity", "ENVIRONMENT", "sensor.outdoor_humidity_sensor"),
    ("Zone Air Temperature", "NORTH PERIMETER", "sensor.north_air_temperature_sensor"),
    ("Zone Air Temperature", "SOUTH PERIMETER", "sensor.south_air_temperature_sensor"),
    ("Zone Air Temperature", "EAST PERIMETER", "sensor.east_air_temperature_sensor"),
    ("Zone Air Temperature", "WEST PERIMETER", "sensor.west_air_temperature_sensor"),
    ("Zone Air Relative Humidity", "NORTH PERIMETER", "sensor.north_air_humidity_sensor"),
    ("Zone Air Relative Humidity", "SOUTH PERIMETER", "sensor.south_air_humidity_sensor"),
    ("Zone Air Relative Humidity", "EAST PERIMETER", "sensor.east_air_humidity_sensor"),
    ("Zone Air Relative Humidity", "WEST PERIMETER", "sensor.west_air_humidity_sensor"),
    ("Zone Radiant HVAC Heating Rate", "LOZARADIANTE_ZONANORTH", "sensor.north_heating_rate_sensor"),
    ("Zone Radiant HVAC Heating Rate", "LOZARADIANTE_ZONASOUTH", "sensor.south_heating_rate_sensor"),
    ("Zone Radiant HVAC Heating Rate", "LOZARADIANTE_ZONAEAST", "sensor.east_heating_rate_sensor"),
    ("Zone Radiant HVAC Heating Rate", "LOZARADIANTE_ZONAWEST", "sensor.west_heating_rate_sensor"),
    ("Heat Pump Electricity Rate", "BOMBACALOR_HP", "sensor.heat_pump_power_sensor"),
]

# total_electricity_hvac: usamos el mismo valor que heat pump por ahora
HA_DUPLICATE = ("sensor.heat_pump_power_sensor", "sensor.total_electricity_hvac_sensor")


def _normalize_csv_header(h: str) -> tuple[str, str]:
    """Extrae variable name y key del encabezado CSV.
    Formatos: 'Key:Variable Name [Unit](Hourly)' o 'Variable Name [Unit](Key)'.
    """
    h = h.strip()
    if not h:
        return "", ""
    # "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)" -> key, var
    if ":" in h:
        key, rest = h.split(":", 1)
        key = key.strip().upper()
        # "Site Outdoor Air Drybulb Temperature [C](Hourly)" -> var name
        m = re.match(r"^(.+?)\s*\[[^\]]*\]\s*\([^)]*\)\s*$", rest.strip())
        var = m.group(1).strip() if m else rest.strip()
        return var, key
    # Fallback: "Variable [Unit](Key)"
    m = re.match(r"^(.+?)\s*\[[^\]]*\]\s*\((.+)\)\s*$", h)
    if m:
        return m.group(1).strip(), m.group(2).strip().upper()
    return h, ""


def _find_column_map(row: list[str]) -> dict[tuple[str, str], int]:
    """Mapa (var_name, key) -> índice de columna."""
    out: dict[tuple[str, str], int] = {}
    for i, cell in enumerate(row):
        v, k = _normalize_csv_header(cell)
        if v:
            out[(v, k)] = i
    return out


def _parse_csv(path: Path) -> list[dict[tuple[str, str], float]]:
    """Lee CSV de ReadVarsESO y devuelve lista de filas como dict (var, key) -> valor."""
    out = _parse_csv_with_datetime(path)
    return [d for _dt, d in out]


def _parse_csv_with_datetime(
    path: Path,
) -> list[tuple[str, dict[tuple[str, str], float]]]:
    """Como _parse_csv pero devuelve (Date/Time, row_dict) por cada fila."""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        headers = next(r, None)
        if not headers:
            return []
        col_map = _find_column_map(headers)
        rows: list[tuple[str, dict[tuple[str, str], float]]] = []
        for row in r:
            if not row:
                continue
            dt_str = row[0].strip() if row else ""
            d: dict[tuple[str, str], float] = {}
            for (v, k), idx in col_map.items():
                if idx < len(row) and row[idx].strip() != "":
                    try:
                        d[(v, k)] = float(row[idx])
                    except ValueError:
                        pass
            if d:
                rows.append((dt_str, d))
        return rows


def _parse_eso_last_timestep(path: Path) -> dict[tuple[str, str], float]:
    """Parsea eplusout.eso y devuelve (var_name, key) -> valor del último timestep."""
    out: dict[tuple[str, str], float] = {}
    var_list: list[tuple[str, str]] = []
    data_start = False
    last_data: list[float] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "End of Data Dictionary" in line:
                data_start = True
                continue
            if "End of Data" in line:
                break
            if not data_start:
                # Dictionary: "2, ..." style, then variable lines
                if "," in line and not line.startswith("1,"):
                    parts = line.split(",")
                    if len(parts) >= 4:
                        try:
                            vname = parts[2].strip()
                            key = parts[3].strip() if len(parts) > 3 else ""
                            var_list.append((vname, key))
                        except Exception:
                            pass
                continue
            # Data rows: timestamp followed by values
            parts = line.split(",")
            if len(parts) > len(var_list):
                try:
                    last_data = [float(x) for x in parts[-len(var_list):]]
                except ValueError:
                    pass

    for i, (v, k) in enumerate(var_list):
        if i < len(last_data):
            out[(v, k)] = last_data[i]
    return out


def extract_sensor_values(run_directory: str | Path) -> dict[str, float]:
    """
    Extrae valores de sensores desde el directorio de salida de EnergyPlus.
    Prioridad: eplusout.csv (ReadVarsESO) si existe; si no, eplusout.eso.

    Devuelve dict entity_id -> valor (float).
    """
    run_dir = Path(run_directory)
    csv_path = run_dir / "eplusout.csv"
    eso_path = run_dir / "eplusout.eso"

    by_key: dict[tuple[str, str], float] = {}

    if csv_path.is_file():
        parsed = _parse_csv_with_datetime(csv_path)
        by_key = parsed[-1][1] if parsed else {}
    elif eso_path.is_file():
        by_key = _parse_eso_last_timestep(eso_path)
    else:
        raise FileNotFoundError(
            f"No se encontró eplusout.csv ni eplusout.eso en {run_dir}"
        )
    return _row_to_sensor_dict(by_key)


def _row_to_sensor_dict(by_key: dict[tuple[str, str], float]) -> dict[str, float]:
    """Convierte un dict (var, key) -> valor en entity_id -> valor."""
    result: dict[str, float] = {}
    for var_name, key, entity_id in EP_TO_HA:
        k = key.upper()
        v = by_key.get((var_name, k)) or by_key.get((var_name, key))
        if v is not None:
            result[entity_id] = v
    if "sensor.heat_pump_power_sensor" in result:
        result["sensor.total_electricity_hvac_sensor"] = result["sensor.heat_pump_power_sensor"]
    return result


def parse_datetime_string(dt_str: str) -> tuple[int, int, int, int] | None:
    """
    Parsea Date/Time del CSV de EnergyPlus (ej. ' 07/21  01:00:00').
    Devuelve (month, day, hour, minute) o None si no se puede parsear.
    """
    if not dt_str or not dt_str.strip():
        return None
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})", dt_str.strip())
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))
    return (month, day, hour, minute)


def extract_sensor_values_all_hours(
    run_directory: str | Path,
) -> list[tuple[str, dict[str, float]]]:
    """
    Extrae sensores por cada hora del CSV.
    Devuelve lista de (Date/Time, dict entity_id -> valor), una entrada por hora.
    """
    run_dir = Path(run_directory)
    csv_path = run_dir / "eplusout.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontro eplusout.csv en {run_dir}")
    rows = _parse_csv_with_datetime(csv_path)
    out: list[tuple[str, dict[str, float]]] = []
    for dt_str, by_key in rows:
        out.append((dt_str, _row_to_sensor_dict(by_key)))
    return out
