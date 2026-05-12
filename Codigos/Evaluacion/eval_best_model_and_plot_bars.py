#!/usr/bin/env python3
"""Evaluate PPO/SAC/TD3 models + 3 baselines and plot global annual bars."""

import argparse
import csv
import glob
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3

import sinergym  # noqa: F401
from sinergym.utils.common import create_environment, import_from_path
from sinergym.utils.wrappers import NormalizeObservation


ROOT = "/workspaces/sinergym"
YEAR_FOR_WEEKDAY = 2000
LAMBDA_T = 25.0
PLOT_YEAR = 2025
PMV_OUTER_BAND = 1.5
ZONE_META = {
    "east": {
        "label": "Este",
        "schedule": "ZonaEast",
        "obs_temp_col": "east_perimeter_air_temperature",
        "obs_hum_col": "east_perimeter_air_humidity",
        "eplus_temp_candidates": ["EAST PERIMETER:Zone Air Temperature [C](Hourly)"],
        "eplus_hum_candidates": ["EAST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        "pmv_col": "pmv_east",
        "color": "#f4a261",
    },
    "west": {
        "label": "Oeste",
        "schedule": "ZonaWest",
        "obs_temp_col": "west_perimeter_air_temperature",
        "obs_hum_col": "west_perimeter_air_humidity",
        "eplus_temp_candidates": ["WEST PERIMETER:Zone Air Temperature [C](Hourly)"],
        "eplus_hum_candidates": ["WEST PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        "pmv_col": "pmv_west",
        "color": "#2ca02c",
    },
    "north": {
        "label": "Norte",
        "schedule": "ZonaNorth",
        "obs_temp_col": "north_perimeter_air_temperature",
        "obs_hum_col": "north_perimeter_air_humidity",
        "eplus_temp_candidates": ["NORTH PERIMETER:Zone Air Temperature [C](Hourly)"],
        "eplus_hum_candidates": ["NORTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        "pmv_col": "pmv_north",
        "color": "#4e79a7",
    },
    "south": {
        "label": "Sur",
        "schedule": "ZonaSouth",
        "obs_temp_col": "south_perimeter_air_temperature",
        "obs_hum_col": "south_perimeter_air_humidity",
        "eplus_temp_candidates": ["SOUTH PERIMETER:Zone Air Temperature [C](Hourly)"],
        "eplus_hum_candidates": ["SOUTH PERIMETER:Zone Air Relative Humidity [%](Hourly)"],
        "pmv_col": "pmv_south",
        "color": "#e15759",
    },
}

PPO_GROUPS = {
    "d+d^2": ["2026-03-11_03-34", "2026-03-11_04-49", "2026-03-11_05-43"],
    "d": ["2026-03-11_06-53", "2026-03-11_17-20", "2026-03-11_18-53"],
    "d^2": ["2026-03-11_07-46", "2026-03-11_20-07", "2026-03-14_05-45"],
}

SAC_GROUPS = {
    "d+d^2": ["2026-03-12_03-21", "2026-03-12_04-10", "2026-03-12_05-38"],
    "d": ["2026-03-13_22-12", "2026-03-13_23-17", "2026-03-14_00-55"],
    "d^2": ["2026-03-12_00-05", "2026-03-12_18-50", "2026-03-12_19-50"],
}

TD3_GROUPS = {
    "d+d^2": ["03-14_21-50", "03-14_18-54", "03-14_20-28"],
    "d": ["03-14_23-17", "03-15_04-19", "03-15_02-52"],
    "d^2": ["2026-03-15_06-40", "2026-03-15_08-23", "2026-03-15_18-45"],    
}

# Lotes de entrenamiento (fecha/hora) para tabla de costos d+d^2 + baselines SP23/24/25.
# Cada id debe aparecer en la lista "d+d^2" del algoritmo correspondiente (PPO_GROUPS, etc.).
TABLA_DD2_LOTES_DEFAULT: tuple[str, ...] = ("2026-03-11_03-34",)
TABLA_BASELINE_LABELS_DEFAULT: tuple[str, ...] = ("BL_sp23", "BL_sp24", "BL_sp25")

MONTH_LABELS_ES = [
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Sep",
    "Oct",
    "Nov",
    "Dic",
]


def _batch_id_matches(requested: str, group_entry: str) -> bool:
    if requested == group_entry:
        return True
    if group_entry and group_entry in requested:
        return True
    if group_entry and requested.endswith(group_entry):
        return True
    return False


def resolve_dd2_labels_for_training_batches(
    batch_ids: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    """Devuelve (label_modelo, algoritmo, id_lote) para reward d+d^2."""
    alg_groups: list[tuple[str, dict[str, list[str]]]] = [
        ("PPO", PPO_GROUPS),
        ("SAC", SAC_GROUPS),
        ("TD3", TD3_GROUPS),
    ]
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for bid in batch_ids:
        for alg, groups in alg_groups:
            lst = groups.get("d+d^2", [])
            for i, entry in enumerate(lst):
                if _batch_id_matches(bid, entry):
                    label = f"{alg}_d+d2_r{i + 1}"
                    if label not in seen:
                        seen.add(label)
                        out.append((label, alg, bid))
                    break
    return out


def save_tabla_costos_dd2_y_baseline(
    output_dir: str,
    points: list,
    baseline_summary_csv: str,
    batch_ids: tuple[str, ...],
    baseline_labels: tuple[str, ...],
    tarifa_json: str,
) -> str | None:
    """CSV: costos anuales y mensuales (tarifa UTE) para modelos d+d^2 + baselines."""
    resolved = resolve_dd2_labels_for_training_batches(batch_ids)
    if not resolved:
        print(
            f"[WARN] Tabla d+d2: ningun modelo resuelto para lotes {batch_ids}. "
            "Revisa que cada id exista en PPO_GROUPS/SAC_GROUPS/TD3_GROUPS['d+d^2']. "
            "Se generan igualmente filas de baselines si aplica."
        )

    by_label = {p.label: p for p in points if getattr(p, "kind", "") == "model"}
    rows_out: list[dict[str, str | float | int]] = []

    mes_cols = [f"costo_mes_{i:02d}" for i in range(1, 13)]

    for label, alg, lote in resolved:
        p = by_label.get(label)
        if not p:
            row: dict[str, str | float | int] = {
                "label": label,
                "kind": "model",
                "algoritmo": alg,
                "lote_entrenamiento": lote,
                "costo_anual_acumulado": "",
                "pmv_desviacion_acumulada_anual": "",
                "status": "missing",
                "nota": "No aparece en resumen_modelos_y_baselines.csv",
            }
            row.update(_monthly_cost_fields(None, False))
            rows_out.append(row)
            continue
        nota_m = ""
        try:
            mon = compute_monthly_cost_from_observations(p.source, tarifa_json)
            mf = _monthly_cost_fields(mon, True)
        except Exception as exc:
            mf = _monthly_cost_fields(None, False)
            nota_m = f" Error costo mensual: {exc}"
        row = {
            "label": label,
            "kind": "model",
            "algoritmo": alg,
            "lote_entrenamiento": lote,
            "costo_anual_acumulado": p.costo_anual,
            "pmv_desviacion_acumulada_anual": p.pmv_dev_acum,
            "status": p.status,
            "nota": nota_m.strip(),
        }
        row.update(mf)
        rows_out.append(row)

    bl_by_label: dict[str, object] = {}
    if os.path.exists(baseline_summary_csv):
        want = set(baseline_labels)
        for bp in load_baseline_points_from_summary_csv(baseline_summary_csv):
            if bp.label in want and bp.status == "ok":
                bl_by_label[bp.label] = bp

    for baseline_label in baseline_labels:
        bl = bl_by_label.get(baseline_label)
        if bl:
            nota_m = ""
            try:
                mon = compute_monthly_cost_from_eplusout(bl.source, tarifa_json)
                mf = _monthly_cost_fields(mon, True)
            except Exception as exc:
                mf = _monthly_cost_fields(None, False)
                nota_m = f" Error costo mensual: {exc}"
            base_nota = f"Setpoint {baseline_label.replace('BL_sp', '')} C"
            row = {
                "label": bl.label,
                "kind": "baseline",
                "algoritmo": "",
                "lote_entrenamiento": "",
                "costo_anual_acumulado": bl.costo_anual,
                "pmv_desviacion_acumulada_anual": bl.pmv_dev_acum,
                "status": bl.status,
                "nota": (base_nota + nota_m).strip(),
            }
            row.update(mf)
            rows_out.append(row)
        else:
            row = {
                "label": baseline_label,
                "kind": "baseline",
                "algoritmo": "",
                "lote_entrenamiento": "",
                "costo_anual_acumulado": "",
                "pmv_desviacion_acumulada_anual": "",
                "status": "missing",
                "nota": f"No encontrado en {baseline_summary_csv}",
            }
            row.update(_monthly_cost_fields(None, False))
            rows_out.append(row)

    if not rows_out:
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tabla_costos_anual_y_mensual_dd2_y_baselines_sp23_sp24_sp25.csv")
    fieldnames = [
        "label",
        "kind",
        "algoritmo",
        "lote_entrenamiento",
        "costo_anual_acumulado",
        *mes_cols,
        "pmv_desviacion_acumulada_anual",
        "status",
        "nota",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)
    return out_path


def save_monthly_cost_bars_dd2_drl_y_baselines(
    output_dir: str,
    points: list,
    baseline_summary_csv: str,
    batch_ids: tuple[str, ...],
    baseline_labels: tuple[str, ...],
    tarifa_json: str,
) -> str | None:
    """Figura 2x2: barras de costo mensual (tarifa UTE) para 1 modelo DRL d+d^2 y 3 baselines."""
    resolved = resolve_dd2_labels_for_training_batches(batch_ids)
    by_label = {p.label: p for p in points if getattr(p, "kind", "") == "model"}

    drl_vals: list[float] | None = None
    drl_title = "Modelo DRL (d+d^2)"
    for label, alg, lote in resolved:
        p = by_label.get(label)
        if not p or p.status != "ok" or not p.source:
            continue
        try:
            mon = compute_monthly_cost_from_observations(p.source, tarifa_json)
            drl_vals = [mon.get(m, 0.0) for m in range(1, 13)]
            drl_title = f"Modelo DRL: {label}"
            break
        except Exception:
            continue

    bl_by_label: dict[str, object] = {}
    if os.path.exists(baseline_summary_csv):
        want = set(baseline_labels)
        for bp in load_baseline_points_from_summary_csv(baseline_summary_csv):
            if bp.label in want and bp.status == "ok":
                bl_by_label[bp.label] = bp

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()
    x = np.arange(12)
    months_es = MONTH_LABELS_ES

    ax0 = axes_flat[0]
    if drl_vals is not None:
        ax0.bar(
            x,
            drl_vals,
            color="#1f77b4",
            width=0.72,
            edgecolor="black",
            linewidth=0.35,
        )
    else:
        ax0.text(
            0.5,
            0.5,
            "Sin datos del modelo en el resumen",
            ha="center",
            va="center",
            transform=ax0.transAxes,
            fontsize=11,
        )
    ax0.set_title(drl_title, fontsize=11)
    ax0.set_xticks(x)
    ax0.set_xticklabels(months_es, rotation=45, ha="right", fontsize=9)
    ax0.set_ylabel("Costo mensual ($)")
    ax0.grid(axis="y", alpha=0.3)

    colors_bl = ["#e67e22", "#27ae60", "#c0392b"]
    for k in range(3):
        ax = axes_flat[k + 1]
        if k < len(baseline_labels):
            blab = baseline_labels[k]
            bl = bl_by_label.get(blab)
            if bl:
                try:
                    mon = compute_monthly_cost_from_eplusout(bl.source, tarifa_json)
                    vals = [mon.get(m, 0.0) for m in range(1, 13)]
                    ax.bar(
                        x,
                        vals,
                        color=colors_bl[k % len(colors_bl)],
                        width=0.72,
                        edgecolor="black",
                        linewidth=0.35,
                    )
                    sp = blab.replace("BL_sp", "")
                    ax.set_title(f"Baseline setpoint {sp} °C ({blab})", fontsize=11)
                except Exception as exc:
                    ax.text(
                        0.5,
                        0.5,
                        str(exc)[:120],
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                    ax.set_title(blab, fontsize=11)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Baseline no encontrado",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.set_title(blab, fontsize=11)
        else:
            ax.axis("off")
        if k < len(baseline_labels):
            ax.set_xticks(x)
            ax.set_xticklabels(months_es, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Costo mensual ($)")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Costo mensual (tarifa UTE): DRL d+d^2 vs baselines",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(
        output_dir,
        "bar_costo_mensual_dd2_drl_y_baselines_sp23_24_25.png",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


@dataclass
class ResultPoint:
    label: str
    kind: str
    source: str
    samples: int
    costo_anual: float
    pmv_dev_acum: float
    status: str
    message: str


def detect_algorithm(model_path: str):
    """Try to infer if the model is PPO, SAC or TD3."""
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            with zf.open("data") as f:
                data = json.loads(f.read())
                cls_name = str(data.get("policy_class", ""))
                if "SAC" in cls_name or "sac" in cls_name:
                    return SAC
                if "TD3" in cls_name or "td3" in cls_name:
                    return TD3
                if "ActorCritic" in cls_name or "PPO" in cls_name or "ppo" in cls_name:
                    return PPO
    except Exception:
        pass

    try:
        PPO.load(model_path, device="cpu")
        return PPO
    except Exception:
        pass
    try:
        TD3.load(model_path, device="cpu")
        return TD3
    except Exception:
        return SAC


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


def detect_active_zones_from_epjson(epjson_path: str) -> list[str]:
    with open(epjson_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    schedules = model.get("Schedule:Compact", {})
    active = []
    for zone, meta in ZONE_META.items():
        sched = schedules.get(meta["schedule"])
        if not sched:
            continue
        values = [
            item.get("field")
            for item in sched.get("data", [])
            if isinstance(item.get("field"), (int, float))
        ]
        if any(float(v) > 0.0 for v in values):
            active.append(zone)
    return active


def compute_metrics_from_observations(obs_csv: str, tarifa_json: str, active_zones: list[str]):
    precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta = load_tarifa(tarifa_json)

    pmv_dev_acum = 0.0
    costo_acum = 0.0
    samples = 0
    if not active_zones:
        raise ValueError("No hay zonas activas para evaluar PMV en observations.csv")

    with open(obs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            month = int(float(r["month"]))
            if month == 0:
                continue
            day = int(float(r["day_of_month"]))
            hour = int(float(r["hour"]))
            power_w = float(r["heat_pump_power"])

            step_dev = 0.0
            for zone in active_zones:
                meta = ZONE_META[zone]
                t = float(r[meta["obs_temp_col"]])
                h = float(r[meta["obs_hum_col"]])
                step_dev += pmv_deviation(pmv_simple(t, h))
            pmv_dev_acum += step_dev

            pr = price_at(
                month, day, hour, precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta
            )
            costo_acum += (power_w / 1000.0) * pr
            samples += 1

    return pmv_dev_acum, costo_acum, samples


def compute_monthly_cost_from_observations(obs_csv: str, tarifa_json: str) -> dict[int, float]:
    """Mes 1..12 -> costo ($) en ese mes; misma regla que compute_metrics_from_observations."""
    precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta = load_tarifa(tarifa_json)
    monthly = {m: 0.0 for m in range(1, 13)}
    if not os.path.exists(obs_csv):
        return monthly
    with open(obs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            month = int(float(r["month"]))
            if month < 1 or month > 12:
                continue
            day = int(float(r["day_of_month"]))
            hour = int(float(r["hour"]))
            power_w = float(r["heat_pump_power"])
            pr = price_at(
                month, day, hour, precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta
            )
            monthly[month] += (power_w / 1000.0) * pr
    return monthly


def compute_monthly_cost_from_eplusout(eplusout_csv: str, tarifa_json: str) -> dict[int, float]:
    """Mes 1..12 -> costo ($) en ese mes; misma regla que compute_metrics_from_eplusout."""
    precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta = load_tarifa(tarifa_json)
    monthly = {m: 0.0 for m in range(1, 13)}
    if not os.path.exists(eplusout_csv):
        return monthly
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
        for r in reader:
            dt = r["Date/Time"].strip().split()
            month, day = dt[0].split("/")
            month_i = int(month)
            if month_i < 1 or month_i > 12:
                continue
            hour = int(dt[1].split(":")[0]) - 1
            if hour < 0:
                hour = 0
            power_w = float(r[power_col])
            pr = price_at(
                month_i,
                int(day),
                hour,
                precio_punta,
                precio_fuera,
                punta_ini,
                punta_fin,
                dias_punta,
            )
            monthly[month_i] += (power_w / 1000.0) * pr
    return monthly


def _monthly_cost_fields(monthly: dict[int, float] | None, filled: bool) -> dict[str, float | str]:
    """Columnas costo_mes_01 .. costo_mes_12."""
    out: dict[str, float | str] = {}
    for i in range(1, 13):
        key = f"costo_mes_{i:02d}"
        if filled and monthly is not None:
            out[key] = monthly.get(i, 0.0)
        else:
            out[key] = ""
    return out


def _pick_first_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"No se encontro columna esperada: {candidates}")


def compute_metrics_from_eplusout(eplusout_csv: str, tarifa_json: str, active_zones: list[str]):
    precio_punta, precio_fuera, punta_ini, punta_fin, dias_punta = load_tarifa(tarifa_json)

    pmv_dev_acum = 0.0
    costo_acum = 0.0
    samples = 0
    if not active_zones:
        raise ValueError("No hay zonas activas para evaluar PMV en eplusout.csv")

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
        zone_cols = {}
        for zone in active_zones:
            meta = ZONE_META[zone]
            t_col = _pick_first_column(cols, meta["eplus_temp_candidates"])
            h_col = _pick_first_column(cols, meta["eplus_hum_candidates"])
            zone_cols[zone] = (t_col, h_col)

        for r in reader:
            dt = r["Date/Time"].strip().split()
            month, day = dt[0].split("/")
            hour = int(dt[1].split(":")[0]) - 1
            if hour < 0:
                hour = 0

            power_w = float(r[power_col])
            step_dev = 0.0
            for zone in active_zones:
                t_col, h_col = zone_cols[zone]
                step_dev += pmv_deviation(pmv_simple(float(r[t_col]), float(r[h_col])))
            pmv_dev_acum += step_dev

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


def compute_comfort_from_eplusout(eplusout_csv: str, active_zones: list[str]) -> dict[str, float]:
    """Calcula % de horas en confort PMV por zona activa desde eplusout.csv."""
    if not active_zones:
        raise ValueError("No hay zonas activas para calcular confort en eplusout.csv")

    counts_total = {z: 0 for z in active_zones}
    counts_in = {z: 0 for z in active_zones}

    with open(eplusout_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        zone_cols = {}
        for zone in active_zones:
            meta = ZONE_META[zone]
            t_col = _pick_first_column(cols, meta["eplus_temp_candidates"])
            h_col = _pick_first_column(cols, meta["eplus_hum_candidates"])
            zone_cols[zone] = (t_col, h_col)

        for r in reader:
            for zone in active_zones:
                t_col, h_col = zone_cols[zone]
                pmv = pmv_simple(float(r[t_col]), float(r[h_col]))
                counts_total[zone] += 1
                if -0.5 <= pmv <= 0.5:
                    counts_in[zone] += 1

    return {
        z: (100.0 * counts_in[z] / counts_total[z] if counts_total[z] else 0.0)
        for z in active_zones
    }


def make_model_sources():
    sources = []
    # Habilitados: PPO, SAC y TD3.
    for alg, groups in (("PPO", PPO_GROUPS), ("SAC", SAC_GROUPS), ("TD3", TD3_GROUPS)):
        for violation, ts_list in groups.items():
            for i, ts in enumerate(ts_list, start=1):
                v_short = violation.replace("^2", "2")
                label = f"{alg}_{v_short}_r{i}"
                sources.append(
                    {
                        "kind": "model",
                        "algorithm": alg,
                        "violation": violation,
                        "timestamp": ts,
                        "label": label,
                    }
                )
    return sources


def resolve_run_dir(algorithm: str, timestamp: str):
    pattern = f"{ROOT}/**/Eplus-{algorithm}-training-nuestroMultizona_*{timestamp}*-res*"
    matches = sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p))
    return matches[0] if matches else None


def build_wrappers():
    return {
        "sinergym.utils.wrappers:NormalizeAction": {},
        "sinergym.utils.wrappers:NormalizeObservation": {},
        "sinergym.utils.wrappers:LoggerWrapper": {
            "storage_class": import_from_path("sinergym.utils.logger:LoggerStorage")
        },
        "sinergym.utils.wrappers:CSVLogger": {},
    }


def evaluate_best_model(run_dir: str, env_id: str, experiment_name: str):
    eval_dir = os.path.join(run_dir, "evaluation")
    model_path = os.path.join(eval_dir, "best_model.zip")
    mean_path = os.path.join(eval_dir, "mean.txt")
    var_path = os.path.join(eval_dir, "var.txt")
    count_path = os.path.join(eval_dir, "count.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe best_model.zip: {model_path}")

    env_name = f"{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    env = create_environment(
        env_id=env_id,
        env_params={"env_name": env_name},
        wrappers=build_wrappers(),
        env_deep_update=True,
    )

    if os.path.exists(mean_path) and os.path.exists(var_path):
        mean = np.loadtxt(mean_path)
        var = np.loadtxt(var_path)
        count = float(np.loadtxt(count_path)) if os.path.exists(count_path) else 1e4

        env_tmp = env
        while env_tmp is not None:
            if isinstance(env_tmp, NormalizeObservation):
                env_tmp.set_mean(mean)
                env_tmp.set_var(var)
                env_tmp.set_count(count)
                env_tmp.deactivate_update()
                break
            env_tmp = getattr(env_tmp, "env", None)

    alg_class = detect_algorithm(model_path)
    model = alg_class.load(model_path, device="cpu")
    model.set_env(env)

    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    workspace = env.get_wrapper_attr("workspace_path")
    env.close()

    obs_csv = None
    for root, dirs, files in os.walk(workspace):
        if "observations.csv" in files:
            obs_csv = os.path.join(root, "observations.csv")
            break
    if obs_csv is None:
        raise FileNotFoundError(f"No se encontro observations.csv dentro de {workspace}")

    return obs_csv, workspace


def _build_datetime_from_obs(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        {
            "year": PLOT_YEAR,
            "month": df["month"].astype(float).astype(int),
            "day": df["day_of_month"].astype(float).astype(int),
            "hour": df["hour"].astype(float).astype(int),
        },
        errors="coerce",
    )


def percent_in_comfort(pmv_values: pd.Series) -> float:
    in_range = ((pmv_values >= -0.5) & (pmv_values <= 0.5)).sum()
    total = len(pmv_values)
    return 100.0 * in_range / total if total else 0.0


def save_model_pmv_background_plot(
    obs_csv: str, label: str, out_dir: str, active_zones: list[str]
) -> tuple[str, dict[str, float]]:
    """Genera grafica PMV exterior/interior para zonas activas de un modelo evaluado."""
    df = pd.read_csv(obs_csv)
    required_cols = [
        "month",
        "day_of_month",
        "hour",
        "outdoor_temperature",
        "outdoor_humidity",
    ]
    for zone in active_zones:
        meta = ZONE_META[zone]
        required_cols.extend([meta["obs_temp_col"], meta["obs_hum_col"]])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en observations.csv: {missing}")

    df["datetime"] = _build_datetime_from_obs(df)
    df = df.dropna(subset=["datetime"]).copy()
    df = df[df["month"].astype(float) > 0].sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay datos validos de observacion para graficar PMV")

    df["pmv_outdoor"] = pmv_simple(df["outdoor_temperature"], df["outdoor_humidity"])
    zone_dev_by_zone = {}
    zone_cfg = []
    for zone in active_zones:
        meta = ZONE_META[zone]
        df[meta["pmv_col"]] = pmv_simple(df[meta["obs_temp_col"]], df[meta["obs_hum_col"]])
        zone_dev_by_zone[zone] = float(df[meta["pmv_col"]].apply(pmv_deviation).sum())
        zone_cfg.append((zone, meta["pmv_col"], f"Zona {meta['label']}", meta["color"]))

    fig, axes = plt.subplots(len(zone_cfg), 1, figsize=(18, 4.5 * len(zone_cfg)), sharex=True)
    if len(zone_cfg) == 1:
        axes = [axes]

    for ax, (zone, zone_col, zone_name, color) in zip(axes, zone_cfg):
        ax.plot(
            df["datetime"],
            df["pmv_outdoor"],
            color="#616161",
            linewidth=0.9,
            alpha=0.9,
            label="PMV clima exterior (hora a hora)",
        )
        ax.plot(
            df["datetime"],
            df[zone_col],
            color=color,
            linewidth=0.9,
            alpha=0.9,
            label=f"PMV interior {zone_name} (hora a hora)",
        )
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(-0.5, color="blue", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_ylabel("PMV")
        ax.set_title(zone_name, loc="left", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
        zone_dev = zone_dev_by_zone[zone]
        ax.text(
            0.995,
            0.90,
            f"Desviacion PMV anual: {zone_dev:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdbdbd", alpha=0.9),
        )

    fig.suptitle(
        f"{label} - PMV interior con PMV de clima exterior (hora a hora)",
        fontsize=16,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Mes")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}_pmv_background.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, zone_dev_by_zone


def save_model_pmv_may_oct_indoor_plot(
    obs_csv: str, label: str, out_dir: str, active_zones: list[str]
) -> str:
    """Genera PMV interior por zona entre 1 mayo y 1 octubre, sin clima exterior."""
    df = pd.read_csv(obs_csv)
    required_cols = ["month", "day_of_month", "hour"]
    for zone in active_zones:
        meta = ZONE_META[zone]
        required_cols.extend([meta["obs_temp_col"], meta["obs_hum_col"]])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en observations.csv: {missing}")

    start_dt = pd.Timestamp(PLOT_YEAR, 5, 1, 0, 0, 0)
    end_dt = pd.Timestamp(PLOT_YEAR, 10, 1, 0, 0, 0)
    df["datetime"] = _build_datetime_from_obs(df)
    df = df.dropna(subset=["datetime"]).copy()
    df = df[df["month"].astype(float) > 0].copy()
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)].copy()
    # Excluye potenciales dias de diseño: para timestamps repetidos conserva el ultimo.
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay datos validos de observacion para mayo-octubre")

    zone_cfg = []
    for zone in active_zones:
        meta = ZONE_META[zone]
        df[meta["pmv_col"]] = pmv_simple(df[meta["obs_temp_col"]], df[meta["obs_hum_col"]])
        zone_cfg.append((zone, meta["pmv_col"], f"Zona {meta['label']}", meta["color"]))

    fig, axes = plt.subplots(len(zone_cfg), 1, figsize=(18, 4.5 * len(zone_cfg)), sharex=True)
    if len(zone_cfg) == 1:
        axes = [axes]

    for ax, (_zone, zone_col, zone_name, color) in zip(axes, zone_cfg):
        ax.plot(
            df["datetime"],
            df[zone_col],
            color=color,
            linewidth=1.8,
            alpha=0.95,
            label=f"PMV interior {zone_name} (hora a hora)",
        )
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(-0.5, color="blue", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(0.0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.set_ylabel("PMV")
        ax.set_title(zone_name, loc="left", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
        ax.set_xlim(start_dt, end_dt)

    fig.suptitle(
        f"{label} - PMV interior (1 mayo a 1 octubre, sin clima exterior)",
        fontsize=16,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Mes")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}_pmv_indoor_may01_oct01.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_model_may_oct_plots(output_dir: str, points, active_zones: list[str]):
    out_dir = os.path.join(output_dir, "plots_pmv_may01_oct01_sin_exterior_algoritmos")
    out_paths = []
    for p in points:
        if p.status != "ok" or p.kind != "model":
            continue
        if not p.source or not os.path.exists(p.source):
            continue
        try:
            out_paths.append(save_model_pmv_may_oct_indoor_plot(p.source, p.label, out_dir, active_zones))
        except Exception as e:
            print(f"[WARN] {p.label}: no se pudo generar PMV mayo-octubre sin exterior: {e}")
    return out_dir, out_paths


def save_zone_pmv_deviation_bars(
    output_dir: str, zone_dev_points: list[tuple[str, dict[str, float]]], active_zones: list[str]
) -> list[str]:
    """Guarda una barra por zona con desviacion PMV acumulada anual por modelo."""
    if not zone_dev_points:
        raise ValueError("No hay puntos de desviacion PMV por zona para graficar.")

    labels = [label for label, _ in zone_dev_points]
    x = np.arange(len(labels))
    fig_w = max(12, 0.70 * max(1, len(labels)))
    out_paths: list[str] = []
    for zone in active_zones:
        meta = ZONE_META[zone]
        dev_values = [zone_map.get(zone, 0.0) for _, zone_map in zone_dev_points]

        fig, ax = plt.subplots(figsize=(fig_w, 6))
        ax.bar(labels, dev_values, color=meta["color"])
        ax.set_ylabel("Desviacion PMV acumulada anual")
        ax.set_title(f"Desviacion PMV acumulada anual - zona {meta['label']} (modelos)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"bar_desviacion_pmv_anual_zona_{zone}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def save_global_bars(output_dir: str, points):
    ok_points = [p for p in points if p.status == "ok"]
    labels = [p.label for p in ok_points]
    costos = [p.costo_anual for p in ok_points]
    pmvs = [p.pmv_dev_acum for p in ok_points]
    rewards = [c + LAMBDA_T * p for c, p in zip(costos, pmvs)]

    # Paletas por metrica:
    # - Costo: tonos frios.
    # - PMV: tonos calidos (como version anterior).
    # Exactamente 9 colores: uno por combinacion algoritmo+reward.
    cool_palette = [
        "#001219",  # very dark navy
        "#003566",  # deep blue
        "#3A86FF",  # vivid blue
        "#4CC9F0",  # bright cyan
        "#00B4D8",  # strong cyan
        "#2A9D8F",  # teal
        "#06D6A0",  # mint/green-cyan
        "#4361EE",  # indigo blue
        "#6C757D",  # slate gray-blue
    ]
    warm_palette = [
        "#7F0000",  # dark red
        "#D00000",  # strong red
        "#E63946",  # vivid red
        "#FF6D00",  # vivid orange
        "#F48C06",  # amber-orange
        "#E76F51",  # coral
        "#EF476F",  # rose
        "#F72585",  # hot pink
        "#B5179E",  # magenta
    ]
    reward_palette = [
        "#3C096C", "#5A189A", "#6A4C93", "#7B2CBF", "#8E44AD",
        "#9D4EDD", "#B388EB", "#C77DFF", "#E0AAFF",
    ]

    def group_key(label: str) -> str:
        # Misma combinacion algoritmo+reward (p.ej. PPO_d2, SAC_d, TD3_d+d2)
        # comparte color en r1/r2/r3.
        if "_r" in label:
            return label.rsplit("_r", 1)[0]
        return label

    def colors_by_group(all_labels, palette):
        def mixed_palette(src):
            # Mezcla fija para 9 colores: separa tonos consecutivos.
            if len(src) == 9:
                order = [0, 4, 8, 3, 7, 2, 6, 1, 5]
                return [src[i] for i in order]
            return src

        palette_mixed = mixed_palette(palette)
        ordered_groups = []
        for lb in all_labels:
            g = group_key(lb)
            if g not in ordered_groups:
                ordered_groups.append(g)
        g_to_color = {
            g: palette_mixed[i % len(palette_mixed)]
            for i, g in enumerate(ordered_groups)
        }
        return [g_to_color[group_key(lb)] for lb in all_labels]

    cost_colors = colors_by_group(labels, cool_palette)
    pmv_colors = colors_by_group(labels, warm_palette)
    reward_colors = colors_by_group(labels, reward_palette)

    def spectrum_colors_by_group(all_labels):
        def algorithm_from_group(group: str) -> str:
            return group.split("_", 1)[0] if "_" in group else "OTHER"

        ordered_groups = []
        for lb in all_labels:
            g = group_key(lb)
            if g not in ordered_groups:
                ordered_groups.append(g)

        # 3 gamas distintas (una por algoritmo) con 3 tonos por reward-group.
        palettes_by_alg = {
            "PPO": ["#0D47A1", "#1976D2", "#64B5F6"],  # azules
            "SAC": ["#1B5E20", "#2E7D32", "#81C784"],  # verdes
            "TD3": ["#B71C1C", "#D32F2F", "#EF9A9A"],  # rojos
            "OTHER": ["#4A4A4A", "#7A7A7A", "#B0B0B0"],
        }
        alg_group_order = {}
        for g in ordered_groups:
            alg = algorithm_from_group(g)
            if alg not in alg_group_order:
                alg_group_order[alg] = []
            alg_group_order[alg].append(g)

        group_to_color = {}
        for alg, groups in alg_group_order.items():
            palette = palettes_by_alg.get(alg, palettes_by_alg["OTHER"])
            for i, g in enumerate(groups):
                group_to_color[g] = palette[i % len(palette)]

        return [group_to_color[group_key(lb)] for lb in all_labels]

    scatter_colors = spectrum_colors_by_group(labels)

    fig_w = max(12, 0.55 * max(1, len(labels)))

    fig1, ax1 = plt.subplots(figsize=(fig_w, 6))
    ax1.bar(labels, costos, color=cost_colors)
    ax1.set_ylabel("Costo anual acumulado ($)")
    ax1.set_title("Costo anual acumulado - modelos RL")
    ax1.tick_params(axis="x", rotation=70)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_cost = os.path.join(output_dir, "bar_costo_anual_acumulado_modelos_y_baselines.png")
    plt.savefig(out_cost, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(fig_w, 6))
    ax2.bar(labels, pmvs, color=pmv_colors)
    ax2.set_ylabel("Desviacion PMV acumulada anual")
    ax2.set_title("Desviacion PMV acumulada anual - modelos RL")
    ax2.tick_params(axis="x", rotation=70)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_pmv = os.path.join(output_dir, "bar_pmv_desviacion_acumulada_modelos_y_baselines.png")
    plt.savefig(out_pmv, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(fig_w, 6))
    ax3.bar(labels, rewards, color=reward_colors)
    ax3.set_ylabel(f"Reward anual (costo + {LAMBDA_T:g} * desviacion PMV)")
    ax3.set_title(f"Reward anual - costo + {LAMBDA_T:g} * desviacion PMV")
    ax3.tick_params(axis="x", rotation=70)
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_reward = os.path.join(output_dir, "bar_reward_anual_costo_mas_pmv.png")
    plt.savefig(out_reward, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    ax4.scatter(pmvs, costos, c=scatter_colors, s=80, alpha=0.9, edgecolors="black", linewidths=0.5)
    def run_index_from_label(label: str) -> int:
        if "_r" in label:
            suffix = label.rsplit("_r", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return 9999

    grouped_points = {}
    for x, y, label, color in zip(pmvs, costos, labels, scatter_colors):
        g = group_key(label)
        if g not in grouped_points:
            grouped_points[g] = {"color": color, "pts": []}
        grouped_points[g]["pts"].append((x, y, label))

    for group_data in grouped_points.values():
        pts = sorted(group_data["pts"], key=lambda p: run_index_from_label(p[2]))
        xy = [(x, y) for x, y, _ in pts]
        if len(xy) >= 2:
            if len(xy) >= 3:
                # Cierra el triángulo: vuelve al primer punto.
                xy = xy + [xy[0]]
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            ax4.plot(xs, ys, color=group_data["color"], linewidth=1.8, alpha=0.9, zorder=1)

    for x, y, label in zip(pmvs, costos, labels):
        ax4.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.9,
        )
    ax4.set_xlabel("Desviacion PMV acumulada anual")
    ax4.set_ylabel("Costo anual acumulado ($)")
    ax4.set_title("Costo anual vs desviacion PMV - modelos RL")
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    out_scatter = os.path.join(output_dir, "scatter_costo_vs_desviacion_pmv_modelos_y_baselines.png")
    plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
    plt.close(fig4)

    return out_cost, out_pmv, out_reward, out_scatter


def save_summary_csv(output_dir: str, points):
    out_csv = os.path.join(output_dir, "resumen_modelos_y_baselines.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "kind",
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
                    p.kind,
                    p.source,
                    p.samples,
                    p.costo_anual,
                    p.pmv_dev_acum,
                    p.status,
                    p.message,
                ]
            )
    return out_csv


def load_points_from_summary_csv(summary_csv: str):
    points = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            points.append(
                ResultPoint(
                    label=r.get("label", ""),
                    kind=r.get("kind", "model"),
                    source=r.get("source", ""),
                    samples=int(float(r.get("samples", 0) or 0)),
                    costo_anual=float(r.get("costo_anual_acumulado", 0.0) or 0.0),
                    pmv_dev_acum=float(r.get("pmv_desviacion_acumulada_anual", 0.0) or 0.0),
                    status=r.get("status", "error"),
                    message=r.get("message", ""),
                )
            )
    return points


def load_baseline_points_from_summary_csv(summary_csv: str):
    points = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            points.append(
                ResultPoint(
                    label=r.get("label", ""),
                    kind="baseline",
                    source=r.get("source", ""),
                    samples=int(float(r.get("samples", 0) or 0)),
                    costo_anual=float(r.get("costo_anual_acumulado", 0.0) or 0.0),
                    pmv_dev_acum=float(r.get("pmv_desviacion_acumulada_anual", 0.0) or 0.0),
                    status=r.get("status", "error"),
                    message=r.get("message", ""),
                )
            )
    return points


def compute_pmv_category_percentages_from_eplusout(
    eplusout_csv: str, active_zones: list[str]
) -> dict[str, float]:
    """Calcula porcentajes PMV desde eplusout.csv para 4 categorias."""
    if not active_zones:
        raise ValueError("No hay zonas activas para calcular porcentajes PMV.")

    count_total = 0
    count_comfort = 0
    count_minus1_1 = 0
    count_minus_outer = 0

    with open(eplusout_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        zone_columns = {}
        for zone in active_zones:
            meta = ZONE_META[zone]
            t_col = _pick_first_column(cols, meta["eplus_temp_candidates"])
            h_col = _pick_first_column(cols, meta["eplus_hum_candidates"])
            zone_columns[zone] = (t_col, h_col)

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

    count_between_1 = max(0, count_minus1_1 - count_comfort)
    count_between_outer = max(0, count_minus_outer - count_minus1_1)
    count_rest = max(0, count_total - count_minus_outer)
    return {
        "comfort_pct": 100.0 * count_comfort / count_total,
        "between_minus1_1_pct": 100.0 * count_between_1 / count_total,
        "between_minus2_2_pct": 100.0 * count_between_outer / count_total,
        "rest_pct": 100.0 * count_rest / count_total,
    }


def save_selected_subset_bars(
    output_dir: str,
    model_points,
    active_zones: list[str],
    baseline_summary_csv: str,
    model_label: str = "PPO_d+d2_r1",
    baseline_labels: tuple[str, ...] = ("BL_sp23", "BL_sp24", "BL_sp25"),
):
    """Guarda barras costo/PMV para un subconjunto fijo modelo+baselines."""
    selected = []
    baseline_points = []
    selected.extend([p for p in model_points if p.status == "ok" and p.label == model_label])

    if os.path.exists(baseline_summary_csv):
        baseline_points = load_baseline_points_from_summary_csv(baseline_summary_csv)
        selected.extend(
            [p for p in baseline_points if p.status == "ok" and p.label in set(baseline_labels)]
        )
    else:
        print(f"[WARN] No existe resumen de baselines para subset: {baseline_summary_csv}")

    order = [model_label, *baseline_labels]
    selected_map = {p.label: p for p in selected}
    selected_ordered = [selected_map[k] for k in order if k in selected_map]
    if not selected_ordered:
        raise ValueError("No se encontraron datos para generar subset solicitado.")

    sub_dir = os.path.join(output_dir, "subset_ppo_dd2_r1_y_baselines_23_24_25")
    os.makedirs(sub_dir, exist_ok=True)
    labels = [p.label for p in selected_ordered]
    costos = [p.costo_anual for p in selected_ordered]
    pmvs = [p.pmv_dev_acum for p in selected_ordered]
    x = np.arange(len(labels))

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.bar(labels, costos, color="#1f77b4")
    ax1.set_ylabel("Costo anual acumulado ($)")
    ax1.set_title("Costo anual acumulado - PPO_d+d2_r1 + BL_sp23/24/25")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_cost = os.path.join(sub_dir, "bar_costo_subset_ppo_r1_y_baselines_23_24_25.png")
    plt.savefig(out_cost, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.bar(labels, pmvs, color="#d62728")
    ax2.set_ylabel("Desviacion PMV acumulada anual")
    ax2.set_title("Desviacion PMV acumulada anual - PPO_d+d2_r1 + BL_sp23/24/25")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_pmv = os.path.join(sub_dir, "bar_pmv_subset_ppo_r1_y_baselines_23_24_25.png")
    plt.savefig(out_pmv, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Barra de confort PMV con el mismo estilo apilado de 4 rangos.
    comfort_vals = []
    between1_vals = []
    between_outer_vals = []
    rest_vals = []
    labels_for_comfort = []
    for p in selected_ordered:
        try:
            if p.kind == "model":
                pct = compute_pmv_category_percentages_from_observations(p.source, active_zones)
            else:
                pct = compute_pmv_category_percentages_from_eplusout(p.source, active_zones)
        except Exception as e:
            print(f"[WARN] {p.label}: no se pudo calcular confort PMV para subset: {e}")
            continue
        labels_for_comfort.append(p.label)
        comfort_vals.append(pct["comfort_pct"])
        between1_vals.append(pct["between_minus1_1_pct"])
        between_outer_vals.append(pct["between_minus2_2_pct"])
        rest_vals.append(pct["rest_pct"])

    out_confort = ""
    if labels_for_comfort:
        x2 = np.arange(len(labels_for_comfort))
        b1 = np.array(comfort_vals, dtype=float)
        b2 = np.array(between1_vals, dtype=float)
        b3 = np.array(between_outer_vals, dtype=float)
        b4 = np.array(rest_vals, dtype=float)
        fig3, ax3 = plt.subplots(figsize=(9, 6.5))
        ax3.bar(x2, b1, color="#2ca02c", width=0.75, label="Confort [-0.5, 0.5]")
        ax3.bar(x2, b2, bottom=b1, color="#ffbf00", width=0.75, label="Entre -1 y 1 (fuera confort)")
        ax3.bar(
            x2,
            b3,
            bottom=b1 + b2,
            color="#d62728",
            width=0.75,
            label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
        )
        ax3.bar(x2, b4, bottom=b1 + b2 + b3, color="#8e44ad", width=0.75, label="Resto")
        for i, (v1, v2, v3, v4) in enumerate(zip(b1, b2, b3, b4)):
            if v1 >= 4:
                ax3.text(i, v1 / 2.0, f"{v1:.1f}%", ha="center", va="center", fontsize=9, rotation=90, color="white", fontweight="bold")
            if v2 >= 4:
                ax3.text(i, v1 + v2 / 2.0, f"{v2:.1f}%", ha="center", va="center", fontsize=9, rotation=90, color="black", fontweight="bold")
            if v3 >= 4:
                ax3.text(i, v1 + v2 + v3 / 2.0, f"{v3:.1f}%", ha="center", va="center", fontsize=9, rotation=90, color="white", fontweight="bold")
            if v4 >= 4:
                ax3.text(i, v1 + v2 + v3 + v4 / 2.0, f"{v4:.1f}%", ha="center", va="center", fontsize=9, rotation=90, color="white", fontweight="bold")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("Porcentaje (%)")
        ax3.set_title("Distribucion PMV - PPO_d+d2_r1 + BL_sp23/24/25 (barras apiladas)")
        ax3.set_xticks(x2)
        ax3.set_xticklabels(labels_for_comfort, rotation=25, ha="right")
        ax3.grid(axis="y", alpha=0.3)
        ax3.legend(loc="upper right")
        plt.tight_layout()
        out_confort = os.path.join(sub_dir, "bar_confort_subset_ppo_r1_y_baselines_23_24_25_stacked.png")
        plt.savefig(out_confort, dpi=150, bbox_inches="tight")
        plt.close(fig3)

    # Scatter costo vs desviacion PMV para el mismo subset.
    out_scatter = ""
    out_scatter_confort = ""
    if selected_ordered:
        selected_labels = {p.label for p in selected_ordered}
        s_models = [p for p in selected_ordered if p.kind == "model"]
        s_baselines = [p for p in selected_ordered if p.kind == "baseline"]

        other_models = [
            p for p in model_points
            if p.status == "ok" and p.label not in selected_labels
        ]
        other_baselines = [
            p for p in baseline_points
            if p.status == "ok" and p.label not in selected_labels
        ]
        fig4, ax4 = plt.subplots(figsize=(8.5, 6.5))
        if other_models:
            ax4.scatter(
                [p.pmv_dev_acum for p in other_models],
                [p.costo_anual for p in other_models],
                c="#1f77b4",
                marker="o",
                s=42,
                alpha=0.20,
                edgecolors="none",
                label="Otros modelos DRL (transparencia)",
            )
        if other_baselines:
            ax4.scatter(
                [p.pmv_dev_acum for p in other_baselines],
                [p.costo_anual for p in other_baselines],
                c="#ff7f0e",
                marker="s",
                s=50,
                alpha=0.22,
                edgecolors="none",
                label="Otros baselines (transparencia)",
            )
        if s_models:
            ax4.scatter(
                [p.pmv_dev_acum for p in s_models],
                [p.costo_anual for p in s_models],
                c="#1f77b4",
                marker="*",
                s=190,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.7,
                label="Modelo DRL seleccionado",
            )
        if s_baselines:
            ax4.scatter(
                [p.pmv_dev_acum for p in s_baselines],
                [p.costo_anual for p in s_baselines],
                c="#ff7f0e",
                marker="o",
                s=120,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6,
                label="Baselines seleccionados",
            )

        baseline_subset = [p for p in selected_ordered if p.kind == "baseline"]
        if len(baseline_subset) >= 2:
            def _sp_num(lb: str) -> int:
                if "sp" in lb:
                    tok = lb.split("sp", 1)[-1]
                    if tok.isdigit():
                        return int(tok)
                return 10**9
            baseline_subset = sorted(baseline_subset, key=lambda p: _sp_num(p.label))
            bl_pmvs = [p.pmv_dev_acum for p in baseline_subset]
            bl_costs = [p.costo_anual for p in baseline_subset]
            ax4.plot(bl_pmvs, bl_costs, color="#ff7f0e", linewidth=2.0, alpha=0.9, label="Linea BL_sp23-24-25")

        for p in selected_ordered:
            ax4.annotate(
                p.label,
                (p.pmv_dev_acum, p.costo_anual),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax4.set_xlabel("Desviacion PMV acumulada anual")
        ax4.set_ylabel("Costo anual acumulado ($)")
        ax4.set_title("Costo vs desviacion PMV - PPO_d+d2_r1 + BL_sp23/24/25")
        ax4.grid(alpha=0.3)
        ax4.legend(loc="best", framealpha=0.95)
        plt.tight_layout()
        out_scatter = os.path.join(sub_dir, "scatter_subset_ppo_r1_y_baselines_23_24_25.png")
        plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
        plt.close(fig4)

        # Scatter costo vs % de tiempo en confort (misma estetica general).
        comfort_cache: dict[str, float] = {}

        def _comfort_pct_for_point(p) -> float | None:
            if p.label in comfort_cache:
                return comfort_cache[p.label]
            try:
                if p.kind == "model":
                    pct = compute_pmv_category_percentages_from_observations(p.source, active_zones)
                else:
                    pct = compute_pmv_category_percentages_from_eplusout(p.source, active_zones)
                value = float(pct["comfort_pct"])
                comfort_cache[p.label] = value
                return value
            except Exception:
                return None

        def _xy_from_points(points_list):
            xs, ys, pts_ok = [], [], []
            for pp in points_list:
                xx = _comfort_pct_for_point(pp)
                if xx is None:
                    continue
                xs.append(xx)
                ys.append(pp.costo_anual)
                pts_ok.append(pp)
            return xs, ys, pts_ok

        fig5, ax5 = plt.subplots(figsize=(8.5, 6.5))
        om_x, om_y, _ = _xy_from_points(other_models)
        ob_x, ob_y, _ = _xy_from_points(other_baselines)
        sm_x, sm_y, sm_pts = _xy_from_points(s_models)
        sb_x, sb_y, sb_pts = _xy_from_points(s_baselines)

        if om_x:
            ax5.scatter(
                om_x,
                om_y,
                c="#1f77b4",
                marker="o",
                s=42,
                alpha=0.20,
                edgecolors="none",
                label="Otros modelos DRL (transparencia)",
            )
        if ob_x:
            ax5.scatter(
                ob_x,
                ob_y,
                c="#ff7f0e",
                marker="s",
                s=50,
                alpha=0.22,
                edgecolors="none",
                label="Otros baselines (transparencia)",
            )
        if sm_x:
            ax5.scatter(
                sm_x,
                sm_y,
                c="#1f77b4",
                marker="*",
                s=190,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.7,
                label="Modelo DRL seleccionado",
            )
        if sb_x:
            ax5.scatter(
                sb_x,
                sb_y,
                c="#ff7f0e",
                marker="o",
                s=120,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6,
                label="Baselines seleccionados",
            )

        baseline_subset2 = [p for p in sb_pts if p.kind == "baseline"]
        if len(baseline_subset2) >= 2:
            def _sp_num2(lb: str) -> int:
                if "sp" in lb:
                    tok = lb.split("sp", 1)[-1]
                    if tok.isdigit():
                        return int(tok)
                return 10**9

            baseline_subset2 = sorted(baseline_subset2, key=lambda p: _sp_num2(p.label))
            bl_x = []
            bl_y = []
            for p in baseline_subset2:
                xx = _comfort_pct_for_point(p)
                if xx is None:
                    continue
                bl_x.append(xx)
                bl_y.append(p.costo_anual)
            if len(bl_x) >= 2:
                ax5.plot(
                    bl_x,
                    bl_y,
                    color="#ff7f0e",
                    linewidth=2.0,
                    alpha=0.9,
                    label="Linea BL_sp23-24-25",
                )

        for p in selected_ordered:
            xx = _comfort_pct_for_point(p)
            if xx is None:
                continue
            ax5.annotate(
                p.label,
                (xx, p.costo_anual),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax5.set_xlabel("Porcentaje de tiempo en confort PMV [-0.5, 0.5] (%)")
        ax5.set_ylabel("Costo anual acumulado ($)")
        ax5.set_title("Costo vs % tiempo en confort - PPO_d+d2_r1 + BL_sp23/24/25")
        ax5.grid(alpha=0.3)
        ax5.legend(loc="best", framealpha=0.95)
        plt.tight_layout()
        out_scatter_confort = os.path.join(
            sub_dir,
            "scatter_subset_costo_vs_confort_pct_ppo_r1_y_baselines_23_24_25.png",
        )
        plt.savefig(out_scatter_confort, dpi=150, bbox_inches="tight")
        plt.close(fig5)

        # Scatter costo vs % de tiempo fuera de confort (100 - confort).
        fig6, ax6 = plt.subplots(figsize=(8.5, 6.5))
        omx_out = [100.0 - v for v in om_x]
        obx_out = [100.0 - v for v in ob_x]
        smx_out = [100.0 - v for v in sm_x]
        sbx_out = [100.0 - v for v in sb_x]

        if omx_out:
            ax6.scatter(
                omx_out,
                om_y,
                c="#1f77b4",
                marker="o",
                s=42,
                alpha=0.20,
                edgecolors="none",
                label="Otros modelos DRL (transparencia)",
            )
        if obx_out:
            ax6.scatter(
                obx_out,
                ob_y,
                c="#ff7f0e",
                marker="s",
                s=50,
                alpha=0.22,
                edgecolors="none",
                label="Otros baselines (transparencia)",
            )
        if smx_out:
            ax6.scatter(
                smx_out,
                sm_y,
                c="#1f77b4",
                marker="*",
                s=190,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.7,
                label="Modelo DRL seleccionado",
            )
        if sbx_out:
            ax6.scatter(
                sbx_out,
                sb_y,
                c="#ff7f0e",
                marker="o",
                s=120,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6,
                label="Baselines seleccionados",
            )

        if len(baseline_subset2) >= 2:
            bl_x_out = [100.0 - xx for xx in bl_x]
            if len(bl_x_out) >= 2:
                ax6.plot(
                    bl_x_out,
                    bl_y,
                    color="#ff7f0e",
                    linewidth=2.0,
                    alpha=0.9,
                    label="Linea BL_sp23-24-25",
                )

        for p in selected_ordered:
            xx = _comfort_pct_for_point(p)
            if xx is None:
                continue
            ax6.annotate(
                p.label,
                (100.0 - xx, p.costo_anual),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax6.set_xlabel("Porcentaje de tiempo fuera de confort PMV [-0.5, 0.5] (%)")
        ax6.set_ylabel("Costo anual acumulado ($)")
        ax6.set_title("Costo vs % tiempo fuera de confort - PPO_d+d2_r1 + BL_sp23/24/25")
        ax6.grid(alpha=0.3)
        ax6.legend(loc="best", framealpha=0.95)
        plt.tight_layout()
        out_scatter_fuera = os.path.join(
            sub_dir,
            "scatter_subset_costo_vs_fuera_confort_pct_ppo_r1_y_baselines_23_24_25.png",
        )
        plt.savefig(out_scatter_fuera, dpi=150, bbox_inches="tight")
        plt.close(fig6)
        print(f"Subset scatter costo vs % fuera confort: {out_scatter_fuera}")

        # Scatter costo vs % de tiempo con PMV fuera de [-1, 1] (PMV > 1 o PMV < -1).
        pmv_outside_1_cache: dict[str, float] = {}

        def _pmv_outside_1_pct_for_point(p) -> float | None:
            if p.label in pmv_outside_1_cache:
                return pmv_outside_1_cache[p.label]
            count_total = 0
            count_outside_1 = 0
            try:
                if p.kind == "model":
                    with open(p.source, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            month = int(float(r["month"]))
                            if month == 0:
                                continue
                            for zone in active_zones:
                                meta = ZONE_META[zone]
                                t = float(r[meta["obs_temp_col"]])
                                h = float(r[meta["obs_hum_col"]])
                                pmv = pmv_simple(t, h)
                                count_total += 1
                                if pmv > 1.0 or pmv < -1.0:
                                    count_outside_1 += 1
                else:
                    with open(p.source, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        cols = reader.fieldnames or []
                        zone_cols = {}
                        for zone in active_zones:
                            meta = ZONE_META[zone]
                            t_col = _pick_first_column(cols, meta["eplus_temp_candidates"])
                            h_col = _pick_first_column(cols, meta["eplus_hum_candidates"])
                            zone_cols[zone] = (t_col, h_col)
                        for r in reader:
                            for zone in active_zones:
                                t_col, h_col = zone_cols[zone]
                                pmv = pmv_simple(float(r[t_col]), float(r[h_col]))
                                count_total += 1
                                if pmv > 1.0 or pmv < -1.0:
                                    count_outside_1 += 1
                if count_total == 0:
                    return None
                value = 100.0 * count_outside_1 / count_total
                pmv_outside_1_cache[p.label] = value
                return value
            except Exception:
                return None

        def _xy_from_points_outside_1(points_list):
            xs, ys, pts_ok = [], [], []
            for pp in points_list:
                xx = _pmv_outside_1_pct_for_point(pp)
                if xx is None:
                    continue
                xs.append(xx)
                ys.append(pp.costo_anual)
                pts_ok.append(pp)
            return xs, ys, pts_ok

        fig7, ax7 = plt.subplots(figsize=(8.5, 6.5))
        omx7, omy7, _ = _xy_from_points_outside_1(other_models)
        obx7, oby7, _ = _xy_from_points_outside_1(other_baselines)
        smx7, smy7, smp7 = _xy_from_points_outside_1(s_models)
        sbx7, sby7, sbp7 = _xy_from_points_outside_1(s_baselines)

        if omx7:
            ax7.scatter(
                omx7,
                omy7,
                c="#1f77b4",
                marker="o",
                s=42,
                alpha=0.20,
                edgecolors="none",
                label="Otros modelos DRL (transparencia)",
            )
        if obx7:
            ax7.scatter(
                obx7,
                oby7,
                c="#ff7f0e",
                marker="s",
                s=50,
                alpha=0.22,
                edgecolors="none",
                label="Otros baselines (transparencia)",
            )
        if smx7:
            ax7.scatter(
                smx7,
                smy7,
                c="#1f77b4",
                marker="*",
                s=190,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.7,
                label="Modelo DRL seleccionado",
            )
        if sbx7:
            ax7.scatter(
                sbx7,
                sby7,
                c="#ff7f0e",
                marker="o",
                s=120,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6,
                label="Baselines seleccionados",
            )

        baseline_subset7 = [p for p in sbp7 if p.kind == "baseline"]
        if len(baseline_subset7) >= 2:
            def _sp_num7(lb: str) -> int:
                if "sp" in lb:
                    tok = lb.split("sp", 1)[-1]
                    if tok.isdigit():
                        return int(tok)
                return 10**9

            baseline_subset7 = sorted(baseline_subset7, key=lambda p: _sp_num7(p.label))
            blx7 = []
            bly7 = []
            for p in baseline_subset7:
                xx = _pmv_outside_1_pct_for_point(p)
                if xx is None:
                    continue
                blx7.append(xx)
                bly7.append(p.costo_anual)
            if len(blx7) >= 2:
                ax7.plot(
                    blx7,
                    bly7,
                    color="#ff7f0e",
                    linewidth=2.0,
                    alpha=0.9,
                    label="Linea BL_sp23-24-25",
                )

        for p in selected_ordered:
            xx = _pmv_outside_1_pct_for_point(p)
            if xx is None:
                continue
            ax7.annotate(
                p.label,
                (xx, p.costo_anual),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax7.set_xlabel("Porcentaje de tiempo con PMV fuera de [-1, 1] (%)")
        ax7.set_ylabel("Costo anual acumulado ($)")
        ax7.set_title("Costo vs % tiempo PMV fuera de [-1,1] - PPO_d+d2_r1 + BL_sp23/24/25")
        ax7.grid(alpha=0.3)
        ax7.legend(loc="best", framealpha=0.95)
        plt.tight_layout()
        out_scatter_gt1 = os.path.join(
            sub_dir,
            "scatter_subset_costo_vs_pmv_fuera_de_menos1_1_pct_ppo_r1_y_baselines_23_24_25.png",
        )
        plt.savefig(out_scatter_gt1, dpi=150, bbox_inches="tight")
        plt.close(fig7)
        print(f"Subset scatter costo vs % PMV fuera de [-1,1]: {out_scatter_gt1}")

    if out_scatter_confort:
        print(f"Subset scatter costo vs % confort: {out_scatter_confort}")

    return sub_dir, out_cost, out_pmv, out_confort, out_scatter


def save_full_models_baselines_bundle(
    output_dir: str,
    model_points,
    active_zones: list[str],
    baseline_summary_csv: str,
):
    """Genera un bundle (costo, PMV, confort, scatter) con todos los modelos y baselines."""
    model_ok = [p for p in model_points if p.status == "ok" and p.kind == "model"]
    baseline_ok = []
    if os.path.exists(baseline_summary_csv):
        baseline_ok = [p for p in load_baseline_points_from_summary_csv(baseline_summary_csv) if p.status == "ok"]
    else:
        print(f"[WARN] No existe resumen de baselines para bundle completo: {baseline_summary_csv}")

    if not model_ok and not baseline_ok:
        raise ValueError("No hay puntos para bundle completo de modelos+baselines.")

    def _sp_num(lb: str) -> int:
        if "sp" in lb:
            tok = lb.split("sp", 1)[-1]
            if tok.isdigit():
                return int(tok)
        return 10**9

    baseline_ok = sorted(baseline_ok, key=lambda p: _sp_num(p.label))
    all_points = model_ok + baseline_ok

    out_dir = os.path.join(output_dir, "bundle_completo_modelos_y_baselines")
    os.makedirs(out_dir, exist_ok=True)

    labels = [p.label for p in all_points]
    costos = [p.costo_anual for p in all_points]
    pmvs = [p.pmv_dev_acum for p in all_points]
    kinds = [p.kind for p in all_points]
    x = np.arange(len(labels))

    # 1) Barras de costo
    fig1, ax1 = plt.subplots(figsize=(max(14, 0.48 * len(labels)), 7))
    color_cost = ["#1f77b4" if k == "model" else "#ff7f0e" for k in kinds]
    ax1.bar(labels, costos, color=color_cost)
    ax1.set_ylabel("Costo anual acumulado ($)")
    ax1.set_title("Costo anual acumulado - todos los modelos + baselines")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=70, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_cost = os.path.join(out_dir, "bar_costo_todos_modelos_y_baselines.png")
    plt.savefig(out_cost, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2) Barras de PMV
    fig2, ax2 = plt.subplots(figsize=(max(14, 0.48 * len(labels)), 7))
    color_pmv = ["#d62728" if k == "model" else "#8c564b" for k in kinds]
    ax2.bar(labels, pmvs, color=color_pmv)
    ax2.set_ylabel("Desviacion PMV acumulada anual")
    ax2.set_title("Desviacion PMV acumulada anual - todos los modelos + baselines")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=70, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_pmv = os.path.join(out_dir, "bar_pmv_todos_modelos_y_baselines.png")
    plt.savefig(out_pmv, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3) Confort PMV apilado (4 rangos)
    comfort_vals = []
    between1_vals = []
    between_outer_vals = []
    rest_vals = []
    labels_ok = []
    kinds_ok = []
    for p in all_points:
        try:
            if p.kind == "model":
                pct = compute_pmv_category_percentages_from_observations(p.source, active_zones)
            else:
                pct = compute_pmv_category_percentages_from_eplusout(p.source, active_zones)
        except Exception as e:
            print(f"[WARN] {p.label}: no se pudo calcular confort PMV para bundle completo: {e}")
            continue
        labels_ok.append(p.label)
        kinds_ok.append(p.kind)
        comfort_vals.append(pct["comfort_pct"])
        between1_vals.append(pct["between_minus1_1_pct"])
        between_outer_vals.append(pct["between_minus2_2_pct"])
        rest_vals.append(pct["rest_pct"])

    if not labels_ok:
        raise ValueError("No se pudo calcular confort PMV para bundle completo.")

    x2 = np.arange(len(labels_ok))
    b1 = np.array(comfort_vals, dtype=float)
    b2 = np.array(between1_vals, dtype=float)
    b3 = np.array(between_outer_vals, dtype=float)
    b4 = np.array(rest_vals, dtype=float)
    fig3, ax3 = plt.subplots(figsize=(max(14, 0.48 * len(labels_ok)), 7.2))
    ax3.bar(x2, b1, color="#2ca02c", width=0.75, label="Confort [-0.5, 0.5]")
    ax3.bar(x2, b2, bottom=b1, color="#ffbf00", width=0.75, label="Entre -1 y 1 (fuera confort)")
    ax3.bar(
        x2,
        b3,
        bottom=b1 + b2,
        color="#d62728",
        width=0.75,
        label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
    )
    ax3.bar(x2, b4, bottom=b1 + b2 + b3, color="#8e44ad", width=0.75, label="Resto")

    for i, (v1, v2, v3, v4) in enumerate(zip(b1, b2, b3, b4)):
        if v1 >= 4:
            ax3.text(i, v1 / 2.0, f"{v1:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
        if v2 >= 4:
            ax3.text(i, v1 + v2 / 2.0, f"{v2:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="black", fontweight="bold")
        if v3 >= 4:
            ax3.text(i, v1 + v2 + v3 / 2.0, f"{v3:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
        if v4 >= 4:
            ax3.text(i, v1 + v2 + v3 + v4 / 2.0, f"{v4:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")

    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Porcentaje (%)")
    ax3.set_title("Distribucion PMV - todos los modelos + baselines (barras apiladas)")
    ax3.set_xticks(x2)
    ax3.set_xticklabels(labels_ok, rotation=70, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend(loc="upper right")
    plt.tight_layout()
    out_confort = os.path.join(out_dir, "bar_confort_todos_modelos_y_baselines_stacked.png")
    plt.savefig(out_confort, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 4) Scatter costo vs PMV (marcando modelos vs baselines)
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    m_points = [p for p in all_points if p.kind == "model"]
    b_points = [p for p in all_points if p.kind == "baseline"]
    if m_points:
        ax4.scatter(
            [p.pmv_dev_acum for p in m_points],
            [p.costo_anual for p in m_points],
            c="#1f77b4",
            marker="o",
            s=58,
            alpha=0.55,
            edgecolors="none",
            label="Modelos DRL",
        )
    if b_points:
        ax4.scatter(
            [p.pmv_dev_acum for p in b_points],
            [p.costo_anual for p in b_points],
            c="#ff7f0e",
            marker="s",
            s=70,
            alpha=0.80,
            edgecolors="black",
            linewidths=0.3,
            label="Baselines",
        )
    ax4.set_xlabel("Desviacion PMV acumulada anual")
    ax4.set_ylabel("Costo anual acumulado ($)")
    ax4.set_title("Costo vs desviacion PMV - todos los modelos + baselines")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="best", framealpha=0.95)
    plt.tight_layout()
    out_scatter = os.path.join(out_dir, "scatter_todos_modelos_y_baselines.png")
    plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
    plt.close(fig4)

    return out_dir, out_cost, out_pmv, out_confort, out_scatter


def compute_zone_deviation_from_observations(
    obs_csv: str, active_zones: list[str]
) -> dict[str, float]:
    df = pd.read_csv(obs_csv)
    zone_dev_by_zone = {}
    for zone in active_zones:
        meta = ZONE_META[zone]
        t_col = meta["obs_temp_col"]
        h_col = meta["obs_hum_col"]
        if t_col not in df.columns or h_col not in df.columns:
            raise ValueError(f"Faltan columnas {t_col}/{h_col} en {obs_csv}")
        pmv_series = pmv_simple(df[t_col], df[h_col])
        zone_dev_by_zone[zone] = float(pd.Series(pmv_series).apply(pmv_deviation).sum())
    return zone_dev_by_zone


def build_zone_dev_points_from_points(points, active_zones: list[str]):
    zone_dev_points: list[tuple[str, dict[str, float]]] = []
    for p in points:
        if p.status != "ok" or p.kind != "model":
            continue
        if not p.source or not os.path.exists(p.source):
            print(f"[WARN] {p.label}: source no disponible para barra por zona: {p.source}")
            continue
        try:
            zone_dev_points.append(
                (p.label, compute_zone_deviation_from_observations(p.source, active_zones))
            )
        except Exception as e:
            print(f"[WARN] {p.label}: no se pudo calcular desviacion por zona: {e}")
    return zone_dev_points


def compute_pmv_category_percentages_from_observations(
    obs_csv: str, active_zones: list[str]
) -> dict[str, float]:
    """Calcula porcentajes PMV agregados (zona-hora) para 4 categorias."""
    if not active_zones:
        raise ValueError("No hay zonas activas para calcular porcentajes PMV.")

    df = pd.read_csv(obs_csv)
    pmv_values = []
    for zone in active_zones:
        meta = ZONE_META[zone]
        t_col = meta["obs_temp_col"]
        h_col = meta["obs_hum_col"]
        if t_col not in df.columns or h_col not in df.columns:
            raise ValueError(f"Faltan columnas {t_col}/{h_col} en {obs_csv}")
        pmv_values.append(np.asarray(pmv_simple(df[t_col], df[h_col]), dtype=float))

    if not pmv_values:
        raise ValueError("No se pudieron construir series PMV por zona.")
    pmv_all = np.concatenate(pmv_values)
    total = int(pmv_all.size)
    if total == 0:
        raise ValueError("No hay muestras PMV para calcular porcentajes.")

    in_comfort = int(np.sum((pmv_all >= -0.5) & (pmv_all <= 0.5)))
    in_minus1_1 = int(np.sum((pmv_all >= -1.0) & (pmv_all <= 1.0)))
    in_minus2_2 = int(np.sum((pmv_all >= -PMV_OUTER_BAND) & (pmv_all <= PMV_OUTER_BAND)))
    in_minus1_1_outside_comfort = max(0, in_minus1_1 - in_comfort)
    in_minus2_2_outside_minus1_1 = max(0, in_minus2_2 - in_minus1_1)
    out_minus2_2 = max(0, total - in_minus2_2)

    return {
        "comfort_pct": 100.0 * in_comfort / total,
        "between_minus1_1_pct": 100.0 * in_minus1_1_outside_comfort / total,
        "between_minus2_2_pct": 100.0 * in_minus2_2_outside_minus1_1 / total,
        "rest_pct": 100.0 * out_minus2_2 / total,
    }


def compute_pmv_category_percentages_by_zone_from_observations(
    obs_csv: str, active_zones: list[str]
) -> dict[str, dict[str, float]]:
    """Calcula porcentajes PMV por zona para 4 categorias."""
    if not active_zones:
        raise ValueError("No hay zonas activas para calcular porcentajes PMV por zona.")

    df = pd.read_csv(obs_csv)
    by_zone = {}
    for zone in active_zones:
        meta = ZONE_META[zone]
        t_col = meta["obs_temp_col"]
        h_col = meta["obs_hum_col"]
        if t_col not in df.columns or h_col not in df.columns:
            raise ValueError(f"Faltan columnas {t_col}/{h_col} en {obs_csv}")

        pmv_vals = np.asarray(pmv_simple(df[t_col], df[h_col]), dtype=float)
        total = int(pmv_vals.size)
        if total == 0:
            raise ValueError(f"No hay muestras PMV para zona {zone} en {obs_csv}")
        in_comfort = int(np.sum((pmv_vals >= -0.5) & (pmv_vals <= 0.5)))
        in_minus1_1 = int(np.sum((pmv_vals >= -1.0) & (pmv_vals <= 1.0)))
        in_minus2_2 = int(np.sum((pmv_vals >= -PMV_OUTER_BAND) & (pmv_vals <= PMV_OUTER_BAND)))
        in_minus1_1_outside_comfort = max(0, in_minus1_1 - in_comfort)
        in_minus2_2_outside_minus1_1 = max(0, in_minus2_2 - in_minus1_1)
        out_minus2_2 = max(0, total - in_minus2_2)
        by_zone[zone] = {
            "comfort_pct": 100.0 * in_comfort / total,
            "between_minus1_1_pct": 100.0 * in_minus1_1_outside_comfort / total,
            "between_minus2_2_pct": 100.0 * in_minus2_2_outside_minus1_1 / total,
            "rest_pct": 100.0 * out_minus2_2 / total,
        }
    return by_zone


def save_comfort_pie_charts(output_dir: str, points, active_zones: list[str]):
    """Genera pie charts promedio por grupo (algoritmo+reward)."""
    ok_points = [p for p in points if p.status == "ok" and p.source and os.path.exists(p.source)]
    if not ok_points:
        raise ValueError("No hay corridas OK con source valido para pie charts de confort.")

    pie_dir = os.path.join(output_dir, "piecharts_confort_pmv_promedio_por_grupo")
    os.makedirs(pie_dir, exist_ok=True)

    def group_key(label: str) -> str:
        if "_r" in label:
            return label.rsplit("_r", 1)[0]
        return label

    labels = [
        "Confort [-0.5, 0.5]",
        "Entre -1 y 1 (fuera confort)",
        f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
        "Resto",
    ]
    colors = ["#2ca02c", "#ffbf00", "#d62728", "#8e44ad"]
    grouped = {}

    for p in ok_points:
        g = group_key(p.label)
        if g not in grouped:
            grouped[g] = []
        grouped[g].append(p)

    ordered_groups = []
    for p in ok_points:
        g = group_key(p.label)
        if g not in ordered_groups:
            ordered_groups.append(g)

    rows = []
    for g in ordered_groups:
        group_points = grouped[g]
        comfort_vals = []
        between_vals = []
        between2_vals = []
        rest_vals = []
        run_labels = []
        for p in group_points:
            pct = compute_pmv_category_percentages_from_observations(p.source, active_zones)
            comfort_vals.append(pct["comfort_pct"])
            between_vals.append(pct["between_minus1_1_pct"])
            between2_vals.append(pct["between_minus2_2_pct"])
            rest_vals.append(pct["rest_pct"])
            run_labels.append(p.label)

        comfort_avg = float(np.mean(comfort_vals))
        between_avg = float(np.mean(between_vals))
        between2_avg = float(np.mean(between2_vals))
        rest_avg = float(np.mean(rest_vals))
        values = [comfort_avg, between_avg, between2_avg, rest_avg]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        )
        ax.set_title(f"{g} - Promedio 3 runs")
        ax.axis("equal")
        out_path = os.path.join(pie_dir, f"pie_confort_pmv_promedio_{g}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rows.append(
            {
                "label": g,
                "runs_count": len(group_points),
                "runs": ",".join(sorted(run_labels)),
                "comfort_pct": comfort_avg,
                "between_minus1_1_pct": between_avg,
                "between_minus2_2_pct": between2_avg,
                "rest_pct": rest_avg,
                "img_path": out_path,
            }
        )

    n = len(rows)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes_flat = np.atleast_1d(axes).reshape(-1)
    for ax, row in zip(axes_flat, rows):
        vals = [
            row["comfort_pct"],
            row["between_minus1_1_pct"],
            row["between_minus2_2_pct"],
            row["rest_pct"],
        ]
        ax.pie(
            vals,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        )
        ax.set_title(row["label"], fontsize=8)
        ax.axis("equal")
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.legend(labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Distribucion PMV promedio por grupo (algoritmo+reward)")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    out_grid = os.path.join(output_dir, "piecharts_confort_pmv_promedio_9_grupos_grid.png")
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(output_dir, "resumen_porcentajes_confort_pmv_promedio_por_grupo.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "runs_count",
                "runs",
                "comfort_pct",
                "between_minus1_1_pct",
                "between_minus2_2_pct",
                "rest_pct",
                "img_path",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["label"],
                    row["runs_count"],
                    row["runs"],
                    row["comfort_pct"],
                    row["between_minus1_1_pct"],
                    row["between_minus2_2_pct"],
                    row["rest_pct"],
                    row["img_path"],
                ]
            )

    # Version en barras de los promedios por grupo (reemplazo visual de piecharts).
    bars_dir = os.path.join(output_dir, "barcharts_confort_pmv_promedio_por_grupo")
    os.makedirs(bars_dir, exist_ok=True)
    bar_paths = []
    for row in rows:
        vals = [
            row["comfort_pct"],
            row["between_minus1_1_pct"],
            row["between_minus2_2_pct"],
            row["rest_pct"],
        ]
        fig_b, ax_b = plt.subplots(figsize=(6, 5))
        x = np.arange(4)
        ax_b.bar(x, vals, color=colors, width=0.65)
        for i, v in enumerate(vals):
            ax_b.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_b.set_ylim(0, 100)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels, rotation=15, ha="right")
        ax_b.set_ylabel("Porcentaje (%)")
        ax_b.set_title(f"{row['label']} - Promedio 3 runs")
        ax_b.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_bar = os.path.join(bars_dir, f"bar_confort_pmv_promedio_{row['label']}.png")
        plt.savefig(out_bar, dpi=150, bbox_inches="tight")
        plt.close(fig_b)
        bar_paths.append(out_bar)

    # Resumen en barras apiladas para los 9 grupos.
    grp_labels = [row["label"] for row in rows]
    grp_comfort = np.array([row["comfort_pct"] for row in rows], dtype=float)
    grp_between = np.array([row["between_minus1_1_pct"] for row in rows], dtype=float)
    grp_between2 = np.array([row["between_minus2_2_pct"] for row in rows], dtype=float)
    grp_rest = np.array([row["rest_pct"] for row in rows], dtype=float)
    xg = np.arange(len(grp_labels))
    fig_g, ax_g = plt.subplots(figsize=(max(10, 0.9 * len(grp_labels)), 6.5))
    ax_g.bar(xg, grp_comfort, color="#2ca02c", width=0.75, label="Confort [-0.5, 0.5]")
    ax_g.bar(
        xg,
        grp_between,
        bottom=grp_comfort,
        color="#ffbf00",
        width=0.75,
        label="Entre -1 y 1 (fuera confort)",
    )
    ax_g.bar(
        xg,
        grp_between2,
        bottom=grp_comfort + grp_between,
        color="#d62728",
        width=0.75,
        label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
    )
    ax_g.bar(
        xg,
        grp_rest,
        bottom=grp_comfort + grp_between + grp_between2,
        color="#8e44ad",
        width=0.75,
        label="Resto",
    )
    for i, (v1, v2, v3, v4) in enumerate(zip(grp_comfort, grp_between, grp_between2, grp_rest)):
        if v1 >= 4:
            ax_g.text(i, v1 / 2.0, f"{v1:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
        if v2 >= 4:
            ax_g.text(i, v1 + v2 / 2.0, f"{v2:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="black", fontweight="bold")
        if v3 >= 4:
            ax_g.text(i, v1 + v2 + v3 / 2.0, f"{v3:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
        if v4 >= 4:
            ax_g.text(i, v1 + v2 + v3 + v4 / 2.0, f"{v4:.1f}%", ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
    ax_g.set_ylim(0, 100)
    ax_g.set_ylabel("Porcentaje (%)")
    ax_g.set_title("Distribucion PMV promedio por grupo (barras apiladas)")
    ax_g.set_xticks(xg)
    ax_g.set_xticklabels(grp_labels, rotation=45, ha="right")
    ax_g.grid(axis="y", alpha=0.3)
    ax_g.legend(loc="upper right")
    plt.tight_layout()
    out_group_bar = os.path.join(output_dir, "bar_confort_pmv_promedio_9_grupos_stacked.png")
    plt.savefig(out_group_bar, dpi=150, bbox_inches="tight")
    plt.close(fig_g)

    # Grafica adicional: 27 corridas en barras apiladas (verde/amarillo/rojo)
    # con etiquetas de porcentaje sobre cada tramo.
    ok_points_runs = [p for p in points if p.status == "ok" and p.source and os.path.exists(p.source)]
    run_labels = []
    run_comfort = []
    run_between = []
    run_between2 = []
    run_rest = []
    for p in ok_points_runs:
        pct = compute_pmv_category_percentages_from_observations(p.source, active_zones)
        run_labels.append(p.label)
        run_comfort.append(pct["comfort_pct"])
        run_between.append(pct["between_minus1_1_pct"])
        run_between2.append(pct["between_minus2_2_pct"])
        run_rest.append(pct["rest_pct"])

    fig_w = max(14, 0.45 * max(1, len(run_labels)))
    fig_bar, ax_bar = plt.subplots(figsize=(fig_w, 7))
    x = np.arange(len(run_labels))
    width = 0.75
    b1 = np.array(run_comfort, dtype=float)
    b2 = np.array(run_between, dtype=float)
    b3 = np.array(run_between2, dtype=float)
    b4 = np.array(run_rest, dtype=float)
    ax_bar.bar(x, b1, color="#2ca02c", width=width, label="Confort [-0.5, 0.5]")
    ax_bar.bar(x, b2, bottom=b1, color="#ffbf00", width=width, label="Entre -1 y 1 (fuera confort)")
    ax_bar.bar(
        x,
        b3,
        bottom=b1 + b2,
        color="#d62728",
        width=width,
        label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
    )
    ax_bar.bar(x, b4, bottom=b1 + b2 + b3, color="#8e44ad", width=width, label="Resto")

    for i, (v1, v2, v3, v4) in enumerate(zip(b1, b2, b3, b4)):
        if v1 >= 4:
            ax_bar.text(
                i,
                v1 / 2.0,
                f"{v1:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                rotation=90,
                color="white",
                fontweight="bold",
            )
        if v2 >= 4:
            ax_bar.text(
                i,
                v1 + v2 / 2.0,
                f"{v2:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                rotation=90,
                color="black",
                fontweight="bold",
            )
        if v3 >= 4:
            ax_bar.text(
                i,
                v1 + v2 + v3 / 2.0,
                f"{v3:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                rotation=90,
                color="white",
                fontweight="bold",
            )
        if v4 >= 4:
            ax_bar.text(
                i,
                v1 + v2 + v3 + v4 / 2.0,
                f"{v4:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                rotation=90,
                color="white",
                fontweight="bold",
            )
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Porcentaje (%)")
    ax_bar.set_title("Distribucion PMV por corrida (barras apiladas)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(run_labels, rotation=70, ha="right")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right")
    plt.tight_layout()
    out_runs_bar = os.path.join(output_dir, "bar_confort_pmv_27_corridas_stacked.png")
    plt.savefig(out_runs_bar, dpi=150, bbox_inches="tight")
    plt.close(fig_bar)

    # Version grouped de 27 corridas con las 4 categorias.
    fig_grouped, ax_grouped = plt.subplots(figsize=(max(14, 0.60 * max(1, len(run_labels))), 7))
    xg2 = np.arange(len(run_labels))
    w = 0.20
    ax_grouped.bar(xg2 - 1.5 * w, b1, color="#2ca02c", width=w, label="Confort [-0.5, 0.5]")
    ax_grouped.bar(xg2 - 0.5 * w, b2, color="#ffbf00", width=w, label="Entre -1 y 1 (fuera confort)")
    ax_grouped.bar(
        xg2 + 0.5 * w,
        b3,
        color="#d62728",
        width=w,
        label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
    )
    ax_grouped.bar(xg2 + 1.5 * w, b4, color="#8e44ad", width=w, label="Resto")
    ax_grouped.set_ylim(0, 100)
    ax_grouped.set_ylabel("Porcentaje (%)")
    ax_grouped.set_title("Distribucion PMV por corrida (barras agrupadas)")
    ax_grouped.set_xticks(xg2)
    ax_grouped.set_xticklabels(run_labels, rotation=70, ha="right")
    ax_grouped.grid(axis="y", alpha=0.3)
    ax_grouped.legend(loc="upper right")
    plt.tight_layout()
    out_runs_bar_grouped = os.path.join(output_dir, "bar_confort_pmv_27_corridas_grouped.png")
    plt.savefig(out_runs_bar_grouped, dpi=150, bbox_inches="tight")
    plt.close(fig_grouped)

    # Graficas adicionales: una barra apilada por corrida para cada zona.
    out_runs_bar_by_zone = []
    for zone in active_zones:
        meta = ZONE_META.get(zone, {"label": zone})
        run_labels_zone = []
        run_comfort_zone = []
        run_between_zone = []
        run_between2_zone = []
        run_rest_zone = []
        for p in ok_points_runs:
            by_zone = compute_pmv_category_percentages_by_zone_from_observations(p.source, active_zones)
            z_pct = by_zone.get(zone)
            if z_pct is None:
                continue
            run_labels_zone.append(p.label)
            run_comfort_zone.append(z_pct["comfort_pct"])
            run_between_zone.append(z_pct["between_minus1_1_pct"])
            run_between2_zone.append(z_pct["between_minus2_2_pct"])
            run_rest_zone.append(z_pct["rest_pct"])

        if not run_labels_zone:
            continue

        fig_w_zone = max(14, 0.45 * max(1, len(run_labels_zone)))
        fig_zone, ax_zone = plt.subplots(figsize=(fig_w_zone, 7))
        x_zone = np.arange(len(run_labels_zone))
        b1_zone = np.array(run_comfort_zone, dtype=float)
        b2_zone = np.array(run_between_zone, dtype=float)
        b3_zone = np.array(run_between2_zone, dtype=float)
        b4_zone = np.array(run_rest_zone, dtype=float)
        ax_zone.bar(x_zone, b1_zone, color="#2ca02c", width=0.75, label="Confort [-0.5, 0.5]")
        ax_zone.bar(
            x_zone,
            b2_zone,
            bottom=b1_zone,
            color="#ffbf00",
            width=0.75,
            label="Entre -1 y 1 (fuera confort)",
        )
        ax_zone.bar(
            x_zone,
            b3_zone,
            bottom=b1_zone + b2_zone,
            color="#d62728",
            width=0.75,
            label=f"Entre -{PMV_OUTER_BAND:g} y {PMV_OUTER_BAND:g} (fuera -1..1)",
        )
        ax_zone.bar(
            x_zone,
            b4_zone,
            bottom=b1_zone + b2_zone + b3_zone,
            color="#8e44ad",
            width=0.75,
            label="Resto",
        )

        for i, (v1, v2, v3, v4) in enumerate(zip(b1_zone, b2_zone, b3_zone, b4_zone)):
            if v1 >= 4:
                ax_zone.text(
                    i, v1 / 2.0, f"{v1:.1f}%", ha="center", va="center",
                    fontsize=9, rotation=90, color="white", fontweight="bold"
                )
            if v2 >= 4:
                ax_zone.text(
                    i, v1 + v2 / 2.0, f"{v2:.1f}%", ha="center", va="center",
                    fontsize=9, rotation=90, color="black", fontweight="bold"
                )
            if v3 >= 4:
                ax_zone.text(
                    i, v1 + v2 + v3 / 2.0, f"{v3:.1f}%", ha="center", va="center",
                    fontsize=9, rotation=90, color="white", fontweight="bold"
                )
            if v4 >= 4:
                ax_zone.text(
                    i, v1 + v2 + v3 + v4 / 2.0, f"{v4:.1f}%", ha="center", va="center",
                    fontsize=9, rotation=90, color="white", fontweight="bold"
                )

        ax_zone.set_ylim(0, 100)
        ax_zone.set_ylabel("Porcentaje (%)")
        ax_zone.set_title(f"Distribucion PMV por corrida - zona {meta['label']} (barras apiladas)")
        ax_zone.set_xticks(x_zone)
        ax_zone.set_xticklabels(run_labels_zone, rotation=70, ha="right")
        ax_zone.grid(axis="y", alpha=0.3)
        ax_zone.legend(loc="upper right")
        plt.tight_layout()
        out_zone_path = os.path.join(output_dir, f"bar_confort_pmv_27_corridas_stacked_zona_{zone}.png")
        plt.savefig(out_zone_path, dpi=150, bbox_inches="tight")
        plt.close(fig_zone)
        out_runs_bar_by_zone.append(out_zone_path)

    return (
        pie_dir,
        out_grid,
        out_csv,
        out_runs_bar,
        out_runs_bar_grouped,
        out_runs_bar_by_zone,
        bars_dir,
        out_group_bar,
    )


def evaluate_models(
    env_id: str,
    tarifa_json: str,
    experiment_name: str,
    max_models: int,
    building_epjson: str,
    output_dir: str,
):
    pmv_plots_dir = os.path.join(output_dir, "plots_pmv_background_algoritmos")
    active_zones = detect_active_zones_from_epjson(building_epjson)
    if not active_zones:
        print("[WARN] No se detectaron zonas activas en epJSON; se usara fallback east,west.")
        active_zones = ["east", "west"]
    print(f"[INFO] Zonas activas detectadas para evaluacion/graficas: {', '.join(active_zones)}")

    points = []
    zone_dev_points: list[tuple[str, dict[str, float]]] = []
    model_sources = make_model_sources()
    if max_models > 0:
        model_sources = model_sources[:max_models]

    for src in model_sources:
        label = src["label"]
        run_dir = resolve_run_dir(src["algorithm"], src["timestamp"])
        if run_dir is None:
            msg = f"Run no encontrada para {src['algorithm']} {src['timestamp']}"
            print(f"[WARN] {label}: {msg}")
            points.append(ResultPoint(label, "model", "", 0, 0.0, 0.0, "missing", msg))
            continue

        try:
            print(f"[INFO] {label}: evaluando {run_dir}")
            obs_csv, workspace = evaluate_best_model(run_dir, env_id, experiment_name)
            pmv_dev, costo, samples = compute_metrics_from_observations(
                obs_csv, tarifa_json, active_zones
            )
            points.append(
                ResultPoint(label, "model", obs_csv, samples, costo, pmv_dev, "ok", workspace)
            )
            try:
                out_plot, zone_dev_by_zone = save_model_pmv_background_plot(
                    obs_csv, label, pmv_plots_dir, active_zones
                )
                zone_dev_points.append((label, zone_dev_by_zone))
                print(f"[OK] {label}: PMV background -> {out_plot}")
            except Exception as plot_err:
                print(f"[WARN] {label}: no se pudo generar grafica PMV background: {plot_err}")
        except Exception as e:
            msg = f"Error evaluando modelo: {e}"
            print(f"[WARN] {label}: {msg}")
            points.append(ResultPoint(label, "model", run_dir, 0, 0.0, 0.0, "error", msg))

    pmv_may_oct_dir, pmv_may_oct_paths = save_model_may_oct_plots(output_dir, points, active_zones)
    for p in pmv_may_oct_paths:
        print(f"[OK] PMV mayo-octubre sin exterior -> {p}")

    return points, zone_dev_points, active_zones, pmv_plots_dir, pmv_may_oct_dir


def generate_plots(output_dir: str, points, zone_dev_points, active_zones, write_summary: bool = True):
    out_csv = save_summary_csv(output_dir, points) if write_summary else ""
    out_cost, out_pmv, out_reward, out_scatter = save_global_bars(output_dir, points)
    out_zone_dev_bars: list[str] = []
    if zone_dev_points:
        out_zone_dev_bars = save_zone_pmv_deviation_bars(output_dir, zone_dev_points, active_zones)
    (
        out_pie_dir,
        out_pie_grid,
        out_pie_csv,
        out_runs_bar,
        out_runs_bar_grouped,
        out_runs_bar_by_zone,
        out_group_bars_dir,
        out_group_bar_summary,
    ) = save_comfort_pie_charts(
        output_dir, points, active_zones
    )
    return (
        out_csv,
        out_cost,
        out_pmv,
        out_reward,
        out_scatter,
        out_zone_dev_bars,
        out_pie_dir,
        out_pie_grid,
        out_pie_csv,
        out_runs_bar,
        out_runs_bar_grouped,
        out_runs_bar_by_zone,
        out_group_bars_dir,
        out_group_bar_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO/SAC/TD3 models and plot global bars."
    )
    parser.add_argument(
        "--env-id",
        default="Eplus-nuestroMultizona-uru-continuous-v1",
        help="Sinergym environment id used for evaluation.",
    )
    parser.add_argument(
        "--tarifa-json",
        default=f"{ROOT}/sinergym/data/tarifas/tarifas_ute.json",
        help="Path to tariff json.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Evaluacion-batch-best-model-barras",
        help="Evaluation run name prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Default: /workspaces/sinergym/batch_eval_<timestamp>",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 means all configured models).",
    )
    parser.add_argument(
        "--building-epjson",
        default=f"{ROOT}/sinergym/data/buildings/idf_multiplesZonas_termostato_ae140bxydeg.epJSON",
        help="Modelo epJSON para detectar zonas activas (ZonaEast/West/North/South).",
    )
    parser.add_argument(
        "--from",
        "--from-existing-dir",
        dest="from_existing_dir",
        default="",
        help=(
            "Si se define, no evalua modelos y solo genera graficas usando "
            "<from>/resumen_modelos_y_baselines.csv"
        ),
    )
    parser.add_argument(
        "--baseline-summary-csv",
        default=f"{ROOT}/baseline_setpoints_ae140bxydeg_north-east-west/resumen_baselines_40.csv",
        help="CSV resumen de baselines para generar subset modelo+baselines.",
    )
    parser.add_argument(
        "--tabla-dd2-lotes",
        nargs="*",
        default=None,
        help=(
            "Lotes de entrenamiento (fecha_hora) para tabla CSV d+d^2 + baseline. "
            f"Por defecto: {TABLA_DD2_LOTES_DEFAULT}. Deben existir en *_GROUPS['d+d^2']."
        ),
    )
    parser.add_argument(
        "--tabla-baseline-labels",
        nargs="*",
        default=None,
        dest="tabla_baseline_labels",
        help=(
            "Etiquetas de baselines en resumen_baselines (ej. BL_sp23 BL_sp24 BL_sp25). "
            f"Por defecto: {TABLA_BASELINE_LABELS_DEFAULT}."
        ),
    )
    args = parser.parse_args()

    if args.from_existing_dir:
        input_dir = args.from_existing_dir
        summary_csv = os.path.join(input_dir, "resumen_modelos_y_baselines.csv")
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(f"No existe resumen para modo solo-graficas: {summary_csv}")

        output_dir = args.output_dir or input_dir
        os.makedirs(output_dir, exist_ok=True)
        active_zones = detect_active_zones_from_epjson(args.building_epjson)
        if not active_zones:
            print("[WARN] No se detectaron zonas activas en epJSON; se usara fallback east,west.")
            active_zones = ["east", "west"]

        points = load_points_from_summary_csv(summary_csv)
        if args.max_models > 0:
            points = points[: args.max_models]
        zone_dev_points = build_zone_dev_points_from_points(points, active_zones)
        (
            out_csv,
            out_cost,
            out_pmv,
            out_reward,
            out_scatter,
            out_zone_dev_bars,
            out_pie_dir,
            out_pie_grid,
            out_pie_csv,
            out_runs_bar,
            out_runs_bar_grouped,
            out_runs_bar_by_zone,
            out_group_bars_dir,
            out_group_bar_summary,
        ) = generate_plots(
            output_dir=output_dir,
            points=points,
            zone_dev_points=zone_dev_points,
            active_zones=active_zones,
            write_summary=False,
        )
        pmv_plots_dir = os.path.join(output_dir, "plots_pmv_background_algoritmos")
        pmv_may_oct_dir, pmv_may_oct_paths = save_model_may_oct_plots(
            output_dir=output_dir, points=points, active_zones=active_zones
        )
        for p in pmv_may_oct_paths:
            print(f"[OK] PMV mayo-octubre sin exterior -> {p}")
        print(f"[INFO] Modo solo-graficas desde: {summary_csv}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = args.output_dir or f"{ROOT}/batch_eval_{ts}"
        os.makedirs(output_dir, exist_ok=True)
        points, zone_dev_points, active_zones, pmv_plots_dir, pmv_may_oct_dir = evaluate_models(
            env_id=args.env_id,
            tarifa_json=args.tarifa_json,
            experiment_name=args.experiment_name,
            max_models=args.max_models,
            building_epjson=args.building_epjson,
            output_dir=output_dir,
        )
        (
            out_csv,
            out_cost,
            out_pmv,
            out_reward,
            out_scatter,
            out_zone_dev_bars,
            out_pie_dir,
            out_pie_grid,
            out_pie_csv,
            out_runs_bar,
            out_runs_bar_grouped,
            out_runs_bar_by_zone,
            out_group_bars_dir,
            out_group_bar_summary,
        ) = generate_plots(
            output_dir=output_dir,
            points=points,
            zone_dev_points=zone_dev_points,
            active_zones=active_zones,
            write_summary=True,
        )

    print("\n=== RESUMEN ===")
    ok_n = len([p for p in points if p.status == "ok"])
    miss_n = len([p for p in points if p.status == "missing"])
    err_n = len([p for p in points if p.status == "error"])
    print(f"Total fuentes: {len(points)} | OK: {ok_n} | Missing: {miss_n} | Error: {err_n}")
    if out_csv:
        print(f"CSV resumen: {out_csv}")
    print(f"Barra costo: {out_cost}")
    print(f"Barra PMV: {out_pmv}")
    print(f"Barra reward anual (costo + PMV): {out_reward}")
    print(f"Scatter costo vs desviacion PMV: {out_scatter}")
    print(f"Pie charts confort PMV (individuales): {out_pie_dir}")
    print(f"Pie charts confort PMV (grilla): {out_pie_grid}")
    print(f"CSV porcentajes confort PMV: {out_pie_csv}")
    print(f"Barcharts confort PMV promedio por grupo: {out_group_bars_dir}")
    print(f"Barras apiladas confort PMV promedio (9 grupos): {out_group_bar_summary}")
    print(f"Barras agrupadas confort PMV (27 corridas): {out_runs_bar_grouped}")
    print(f"Barras apiladas confort PMV (27 corridas): {out_runs_bar}")
    if out_runs_bar_by_zone:
        print("Barras apiladas confort PMV por zona (27 corridas):")
        for p in out_runs_bar_by_zone:
            print(f" - {p}")
    if out_zone_dev_bars:
        print("Barras desviacion PMV anual por zona:")
        for p in out_zone_dev_bars:
            print(f" - {p}")
    print(f"Graficas PMV background por algoritmo: {pmv_plots_dir}")
    print(f"Graficas PMV 1 mayo-1 octubre sin exterior (algoritmos): {pmv_may_oct_dir}")
    batch_ids_tabla = tuple(args.tabla_dd2_lotes) if args.tabla_dd2_lotes else TABLA_DD2_LOTES_DEFAULT
    baseline_labels_tabla = (
        tuple(args.tabla_baseline_labels)
        if args.tabla_baseline_labels
        else TABLA_BASELINE_LABELS_DEFAULT
    )
    tabla_dd2_path = save_tabla_costos_dd2_y_baseline(
        output_dir=output_dir,
        points=points,
        baseline_summary_csv=args.baseline_summary_csv,
        batch_ids=batch_ids_tabla,
        baseline_labels=baseline_labels_tabla,
        tarifa_json=args.tarifa_json,
    )
    if tabla_dd2_path:
        print(f"Tabla costos anual d+d^2 + baselines: {tabla_dd2_path}")
    try:
        out_mes_bars = save_monthly_cost_bars_dd2_drl_y_baselines(
            output_dir=output_dir,
            points=points,
            baseline_summary_csv=args.baseline_summary_csv,
            batch_ids=batch_ids_tabla,
            baseline_labels=baseline_labels_tabla,
            tarifa_json=args.tarifa_json,
        )
        if out_mes_bars:
            print(f"Grafica costo mensual DRL + baselines: {out_mes_bars}")
    except Exception as mes_plot_err:
        print(f"[WARN] No se pudo generar grafica costo mensual DRL+baselines: {mes_plot_err}")
    try:
        subset_dir, subset_cost, subset_pmv, subset_confort, subset_scatter = save_selected_subset_bars(
            output_dir=output_dir,
            model_points=points,
            active_zones=active_zones,
            baseline_summary_csv=args.baseline_summary_csv,
        )
        print(f"Subset modelo+baselines (carpeta): {subset_dir}")
        print(f"Subset barra costo: {subset_cost}")
        print(f"Subset barra PMV: {subset_pmv}")
        if subset_confort:
            print(f"Subset barra confort PMV: {subset_confort}")
        if subset_scatter:
            print(f"Subset scatter costo vs PMV: {subset_scatter}")
    except Exception as subset_err:
        print(f"[WARN] No se pudo generar subset PPO_d+d2_r1 + BL_sp23/24/25: {subset_err}")

    try:
        full_dir, full_cost, full_pmv, full_confort, full_scatter = save_full_models_baselines_bundle(
            output_dir=output_dir,
            model_points=points,
            active_zones=active_zones,
            baseline_summary_csv=args.baseline_summary_csv,
        )
        print(f"Bundle completo modelos+baselines (carpeta): {full_dir}")
        print(f"Bundle completo barra costo: {full_cost}")
        print(f"Bundle completo barra PMV: {full_pmv}")
        print(f"Bundle completo barra confort PMV: {full_confort}")
        print(f"Bundle completo scatter costo vs PMV: {full_scatter}")
    except Exception as full_err:
        print(f"[WARN] No se pudo generar bundle completo modelos+baselines: {full_err}")


if __name__ == "__main__":
    main()
