"""Compara PMV aproximado vs PMV real (ASHRAE 55).

Uso rapido:
    python scripts/compare_pmv_approx_vs_ashrae55.py

Tambien se puede personalizar el grid:
    python scripts/compare_pmv_approx_vs_ashrae55.py --t-min 15 --t-max 32 --t-step 0.5 --rh-min 20 --rh-max 95 --rh-step 1
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pythermalcomfort.models import pmv_ppd_ashrae


@dataclass
class ComparisonStats:
    samples: int
    mae: float
    rmse: float
    max_abs_error: float
    mean_signed_error: float
    p_within_01: float
    p_within_025: float
    p_within_05: float
    band_match_05: float
    band_match_1: float
    band_match_2: float
    tp_05: int
    tn_05: int
    fp_05: int
    fn_05: int
    tp_1: int
    tn_1: int
    fp_1: int
    fn_1: int
    tp_2: int
    tn_2: int
    fp_2: int
    fn_2: int
    range1_samples: int
    range2_samples: int
    range3_samples: int
    range1_error_pct: float
    range2_error_pct: float
    range3_error_pct: float


def pmv_simple(tdb: float | np.ndarray, rh: float | np.ndarray) -> float | np.ndarray:
    """Formula aproximada usada en reward."""
    return -7.4928 + 0.2882 * tdb - 0.0020 * rh + 0.0004 * tdb * rh


def _extract_pmv(result: Any) -> float:
    """Extrae PMV del retorno de pythermalcomfort de forma robusta."""
    if isinstance(result, dict):
        return float(result["pmv"])
    if hasattr(result, "pmv"):
        return float(result.pmv)
    raise TypeError(f"No se pudo extraer PMV del tipo de retorno: {type(result)}")


def pmv_ashrae_real(tdb: float, rh: float, vr: float, met: float, clo: float, wme: float) -> float:
    result = pmv_ppd_ashrae(tdb=tdb, tr=tdb, vr=vr, rh=rh, met=met, clo=clo, wme=wme)
    return _extract_pmv(result)


def in_band(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (value >= lo) & (value <= hi)


def confusion_counts(pred_in: np.ndarray, real_in: np.ndarray) -> tuple[int, int, int, int]:
    """Devuelve TP, TN, FP, FN considerando 'in-band' como clase positiva."""
    tp = int(np.sum(pred_in & real_in))
    tn = int(np.sum((~pred_in) & (~real_in)))
    fp = int(np.sum(pred_in & (~real_in)))
    fn = int(np.sum((~pred_in) & real_in))
    return tp, tn, fp, fn


def classify_three_ranges(pmv: np.ndarray) -> np.ndarray:
    """Clasifica PMV en 3 rangos disjuntos:
    1) [-0.5, 0.5]
    2) (-1, 1) fuera de rango 1
    3) (-2, 2) fuera de rango 2
    0) fuera de [-2, 2]
    """
    classes = np.zeros_like(pmv, dtype=np.int8)
    classes[(pmv >= -0.5) & (pmv <= 0.5)] = 1
    classes[((pmv > -1.0) & (pmv < 1.0)) & (classes == 0)] = 2
    classes[((pmv > -2.0) & (pmv < 2.0)) & (classes == 0)] = 3
    return classes


def range_error_pct(real_cls: np.ndarray, approx_cls: np.ndarray, range_id: int) -> tuple[int, float]:
    """Error de clasificacion dentro de un rango:
    sobre las muestras con rango REAL=range_id, % con rango aproximado distinto.
    """
    mask = real_cls == range_id
    n = int(np.sum(mask))
    if n == 0:
        return 0, 0.0
    err_pct = float(np.mean(approx_cls[mask] != real_cls[mask]) * 100.0)
    return n, err_pct


def compute_stats(pmv_approx: np.ndarray, pmv_real: np.ndarray) -> ComparisonStats:
    err = pmv_approx - pmv_real
    abs_err = np.abs(err)

    band_05_approx = in_band(pmv_approx, -0.5, 0.5)
    band_05_real = in_band(pmv_real, -0.5, 0.5)
    band_1_approx = in_band(pmv_approx, -1.0, 1.0)
    band_1_real = in_band(pmv_real, -1.0, 1.0)
    band_2_approx = in_band(pmv_approx, -2.0, 2.0)
    band_2_real = in_band(pmv_real, -2.0, 2.0)
    tp_05, tn_05, fp_05, fn_05 = confusion_counts(band_05_approx, band_05_real)
    tp_1, tn_1, fp_1, fn_1 = confusion_counts(band_1_approx, band_1_real)
    tp_2, tn_2, fp_2, fn_2 = confusion_counts(band_2_approx, band_2_real)
    real_cls = classify_three_ranges(pmv_real)
    approx_cls = classify_three_ranges(pmv_approx)
    range1_n, range1_err = range_error_pct(real_cls, approx_cls, 1)
    range2_n, range2_err = range_error_pct(real_cls, approx_cls, 2)
    range3_n, range3_err = range_error_pct(real_cls, approx_cls, 3)

    return ComparisonStats(
        samples=int(err.size),
        mae=float(np.mean(abs_err)),
        rmse=float(math.sqrt(float(np.mean(err**2)))),
        max_abs_error=float(np.max(abs_err)),
        mean_signed_error=float(np.mean(err)),
        p_within_01=float(np.mean(abs_err <= 0.10) * 100.0),
        p_within_025=float(np.mean(abs_err <= 0.25) * 100.0),
        p_within_05=float(np.mean(abs_err <= 0.50) * 100.0),
        band_match_05=float(np.mean(band_05_approx == band_05_real) * 100.0),
        band_match_1=float(np.mean(band_1_approx == band_1_real) * 100.0),
        band_match_2=float(np.mean(band_2_approx == band_2_real) * 100.0),
        tp_05=tp_05,
        tn_05=tn_05,
        fp_05=fp_05,
        fn_05=fn_05,
        tp_1=tp_1,
        tn_1=tn_1,
        fp_1=fp_1,
        fn_1=fn_1,
        tp_2=tp_2,
        tn_2=tn_2,
        fp_2=fp_2,
        fn_2=fn_2,
        range1_samples=range1_n,
        range2_samples=range2_n,
        range3_samples=range3_n,
        range1_error_pct=range1_err,
        range2_error_pct=range2_err,
        range3_error_pct=range3_err,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compara PMV aproximado (formula reward) vs PMV real ASHRAE 55 (pythermalcomfort)."
    )
    p.add_argument("--t-min", type=float, default=15.0, help="Temperatura minima del grid [C].")
    p.add_argument("--t-max", type=float, default=32.0, help="Temperatura maxima del grid [C].")
    p.add_argument("--t-step", type=float, default=0.5, help="Paso de temperatura [C].")
    p.add_argument("--rh-min", type=float, default=20.0, help="Humedad relativa minima [%].")
    p.add_argument("--rh-max", type=float, default=95.0, help="Humedad relativa maxima [%].")
    p.add_argument("--rh-step", type=float, default=1.0, help="Paso de humedad relativa [%].")
    p.add_argument("--vr", type=float, default=0.1, help="Velocidad relativa del aire [m/s].")
    p.add_argument("--met", type=float, default=1.2, help="Metabolismo [met].")
    p.add_argument("--clo", type=float, default=0.57, help="Aislamiento de ropa [clo].")
    p.add_argument("--wme", type=float, default=0.0, help="Trabajo mecanico externo [met].")
    p.add_argument(
        "--out",
        type=str,
        default="pmv_approx_vs_ashrae55_report.txt",
        help="Ruta del archivo de salida para guardar el reporte.",
    )
    p.add_argument(
        "--out-table",
        type=str,
        default="pmv_approx_vs_ashrae55_values.csv",
        help="Ruta del CSV con todos los valores (aprox vs real).",
    )
    return p


def build_report(args: argparse.Namespace, stats: ComparisonStats) -> str:
    def pct(n: int, d: int) -> float:
        return (100.0 * n / d) if d > 0 else 0.0

    lines = [
        "=" * 78,
        "Comparacion PMV aproximado vs PMV real (ASHRAE 55)",
        "=" * 78,
        (
            f"Grid: T[{args.t_min},{args.t_max}] step={args.t_step} | "
            f"RH[{args.rh_min},{args.rh_max}] step={args.rh_step}"
        ),
        f"Parametros ASHRAE: vr={args.vr}, met={args.met}, clo={args.clo}, wme={args.wme}",
        f"Muestras totales: {stats.samples}",
        "-" * 78,
        f"MAE               : {stats.mae:.4f} PMV",
        f"RMSE              : {stats.rmse:.4f} PMV",
        f"Error max abs     : {stats.max_abs_error:.4f} PMV",
        f"Error medio firmado: {stats.mean_signed_error:+.4f} PMV",
        "-" * 78,
        f"% |error| <= 0.10 : {stats.p_within_01:6.2f}%",
        f"% |error| <= 0.25 : {stats.p_within_025:6.2f}%",
        f"% |error| <= 0.50 : {stats.p_within_05:6.2f}%",
        "-" * 78,
        f"Coincidencia banda [-0.5, 0.5] : {stats.band_match_05:6.2f}%",
        f"Coincidencia banda [-1, 1] : {stats.band_match_1:6.2f}%",
        f"Coincidencia banda [-2, 2] : {stats.band_match_2:6.2f}%",
        "-" * 78,
        "Error de clasificacion dentro de 3 rangos disjuntos (usando PMV real como base):",
        (
            f"Rango 1 [-0.5, 0.5]                  : {stats.range1_error_pct:6.2f}% "
            f"(n={stats.range1_samples})"
        ),
        (
            f"Rango 2 (-1, 1) fuera de [-0.5,0.5]  : {stats.range2_error_pct:6.2f}% "
            f"(n={stats.range2_samples})"
        ),
        (
            f"Rango 3 (-2, 2) fuera de (-1,1)      : {stats.range3_error_pct:6.2f}% "
            f"(n={stats.range3_samples})"
        ),
        "-" * 78,
        "Desglose de clasificacion por banda (in-band = positivo):",
        f"[-0.5, 0.5] -> TP={stats.tp_05} ({pct(stats.tp_05, stats.samples):5.2f}%), "
        f"TN={stats.tn_05} ({pct(stats.tn_05, stats.samples):5.2f}%), "
        f"FP={stats.fp_05} ({pct(stats.fp_05, stats.samples):5.2f}%), "
        f"FN={stats.fn_05} ({pct(stats.fn_05, stats.samples):5.2f}%)",
        f"[-1, 1]     -> TP={stats.tp_1} ({pct(stats.tp_1, stats.samples):5.2f}%), "
        f"TN={stats.tn_1} ({pct(stats.tn_1, stats.samples):5.2f}%), "
        f"FP={stats.fp_1} ({pct(stats.fp_1, stats.samples):5.2f}%), "
        f"FN={stats.fn_1} ({pct(stats.fn_1, stats.samples):5.2f}%)",
        f"[-2, 2]     -> TP={stats.tp_2} ({pct(stats.tp_2, stats.samples):5.2f}%), "
        f"TN={stats.tn_2} ({pct(stats.tn_2, stats.samples):5.2f}%), "
        f"FP={stats.fp_2} ({pct(stats.fp_2, stats.samples):5.2f}%), "
        f"FN={stats.fn_2} ({pct(stats.fn_2, stats.samples):5.2f}%)",
        "=" * 78,
    ]
    return "\n".join(lines) + "\n"


def _range_label_3bands(pmv: float) -> str:
    if -0.5 <= pmv <= 0.5:
        return "R1_-0.5_0.5"
    if -1.0 < pmv < 1.0:
        return "R2_-1_1_fuera_R1"
    if -2.0 < pmv < 2.0:
        return "R3_-2_2_fuera_R2"
    return "R4_fuera_-2_2"


def write_values_table_csv(
    out_csv: Path,
    tt: np.ndarray,
    hh: np.ndarray,
    pmv_approx: np.ndarray,
    pmv_real: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tdb_c",
                "rh_pct",
                "pmv_approx",
                "pmv_real",
                "error_signed",
                "error_abs",
                "rango_real_3bandas",
                "rango_aprox_3bandas",
            ]
        )
        for t, rh, pa, pr in zip(
            tt.ravel(order="C"),
            hh.ravel(order="C"),
            pmv_approx.ravel(order="C"),
            pmv_real.ravel(order="C"),
        ):
            err = float(pa - pr)
            writer.writerow(
                [
                    f"{float(t):.2f}",
                    f"{float(rh):.2f}",
                    f"{float(pa):.6f}",
                    f"{float(pr):.6f}",
                    f"{err:.6f}",
                    f"{abs(err):.6f}",
                    _range_label_3bands(float(pr)),
                    _range_label_3bands(float(pa)),
                ]
            )


def main() -> None:
    args = build_parser().parse_args()

    t_values = np.arange(args.t_min, args.t_max + args.t_step / 2.0, args.t_step, dtype=float)
    rh_values = np.arange(args.rh_min, args.rh_max + args.rh_step / 2.0, args.rh_step, dtype=float)

    tt, hh = np.meshgrid(t_values, rh_values, indexing="ij")
    pmv_approx = np.asarray(pmv_simple(tt, hh), dtype=float)

    pmv_real = np.zeros_like(pmv_approx, dtype=float)
    for i in range(tt.shape[0]):
        for j in range(tt.shape[1]):
            pmv_real[i, j] = pmv_ashrae_real(
                tdb=float(tt[i, j]),
                rh=float(hh[i, j]),
                vr=float(args.vr),
                met=float(args.met),
                clo=float(args.clo),
                wme=float(args.wme),
            )

    stats = compute_stats(pmv_approx=pmv_approx, pmv_real=pmv_real)

    report = build_report(args=args, stats=stats)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    write_values_table_csv(
        out_csv=Path(args.out_table),
        tt=tt,
        hh=hh,
        pmv_approx=pmv_approx,
        pmv_real=pmv_real,
    )


if __name__ == "__main__":
    main()
