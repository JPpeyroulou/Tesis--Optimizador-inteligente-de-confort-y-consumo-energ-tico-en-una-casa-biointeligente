# `compare_pmv_approx_vs_ashrae55.py`

Herramienta **offline** que compara el **PMV aproximado** usado en recompensas / métricas del proyecto:

```text
PMV_aprox = -7.4928 + 0.2882*T_db - 0.0020*RH + 0.0004*T_db*RH
```

frente al **PMV de referencia** calculado con **`pythermalcomfort`**: `pmv_ppd_ashrae` (modelo alineado con **ASHRAE 55**), sobre una **malla regular** de temperatura de bulbo seco y humedad relativa.

No usa EnergyPlus ni Sinergym; solo NumPy y la librería de confort térmico.

## Requisitos

```bash
pip install pythermalcomfort numpy
```

(o el entorno del proyecto si ya incluye `pythermalcomfort`).

## Uso rápido

Desde la raíz del repo:

```bash
python scripts/compare_pmv_approx_vs_ashrae55.py
```

Esto genera, en el directorio de trabajo actual (salvo que cambies `--out` / `--out-table`):

- **`pmv_approx_vs_ashrae55_report.txt`**: texto con MAE, RMSE, error máximo, porcentajes de error por umbrales, coincidencia en bandas [-0,5, 0,5], [-1, 1], [-2, 2], matrices de confusión por banda y errores de clasificación en tres rangos disjuntos.
- **`pmv_approx_vs_ashrae55_values.csv`**: una fila por par (T, RH) con columnas `tdb_c`, `rh_pct`, `pmv_approx`, `pmv_real`, errores y etiquetas de rango.

## Grid y parámetros por defecto

| Parámetro | Default | Rol |
|-----------|---------|-----|
| `--t-min`, `--t-max`, `--t-step` | 15, 32, 0.5 °C | Rango y paso de **T_db** |
| `--rh-min`, `--rh-max`, `--rh-step` | 20, 95, 1 % | Rango y paso de **HR** |
| `--vr` | 0.1 m/s | Velocidad relativa del aire (`pmv_ppd_ashrae`) |
| `--met` | 1.2 met | Metabolismo |
| `--clo` | 0.57 clo | Ropa |
| `--wme` | 0 | Trabajo mecánico externo |

En el modelo de referencia se usa **`tr = tdb`** (temperatura radiante igual a la del aire).

Con los valores por defecto del script hay **2660** puntos (35 temperaturas × 76 humedades relativas).

## Personalizar el experimento

Ejemplo de malla más fina o distinto rango:

```bash
python scripts/compare_pmv_approx_vs_ashrae55.py \
  --t-min 18 --t-max 30 --t-step 0.25 \
  --rh-min 30 --rh-max 80 --rh-step 2 \
  --met 1.1 --clo 0.5 \
  --out informes/pmv_informe.txt \
  --out-table informes/pmv_malla.csv
```

Los directorios de salida se crean si no existen.

## Qué interpretar del informe

- **MAE / RMSE / error máximo:** en **unidades PMV** (no son porcentajes salvo que los conviertas explícitamente).
- **Porcentajes** `|error| ≤ 0.10`, etc.: fracción de la malla bajo ese umbral.
- **Coincidencia de banda:** porcentaje de celdas donde aproximación y referencia **caen a la vez** dentro de la misma banda PMV.
- **TP/TN/FP/FN:** confusión al tratar “estar en banda” como clase positiva (ver código `confusion_counts`).
- **Rangos 1–3:** clasificación en tres franjas disjuntas (núcleo ±0,5, anillos hasta ±1 y ±2); los porcentajes de error miden discordancia entre clasificación por PMV real vs PMV aproximado.

## Coherencia con el código de entrenamiento / evaluación

La fórmula `pmv_simple` es la misma idea que en `eval_best_model_and_plot_bars.py` y `eval_baselines_40_and_plot_bars.py`. Este script sirve para **cuantificar** el desvío frente a un PMV normativo ASHRAE vía `pythermalcomfort`, **no** sustituye al cálculo usado en el reward durante la simulación.
