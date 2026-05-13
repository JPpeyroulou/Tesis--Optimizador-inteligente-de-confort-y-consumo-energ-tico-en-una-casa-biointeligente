# `eval_baselines_40_and_plot_bars.py`

Script que **no ejecuta EnergyPlus**: lee salidas ya simuladas (`eplusout.csv`) de **baselines por setpoint** (carpetas `sp_XX`), calcula **costo anual** con tarifa UTE, **desviación PMV acumulada** con la misma aproximación que el resto del proyecto (`pmv_simple` + `pmv_deviation`), y genera **CSV + figuras** (barras globales, barras apiladas de confort, series PMV mayo–octubre).

## Requisitos

- Python con `numpy`, `matplotlib` (backend `Agg`).
- Archivos `eplusout.csv` en la estructura esperada (ver más abajo).
- JSON de tarifa compatible con `load_tarifa()` (precios punta/fuera, horarios, días punta).

**Nota:** al inicio del script, `ROOT = "/workspaces/sinergym"` está **fijado**. Si el clon vive en otra ruta, ajusta esa constante o pasa rutas absolutas en los argumentos.

## Cómo ejecutarlo

Desde la raíz del repositorio (o ajustando rutas):

```bash
python scripts/eval_baselines_40_and_plot_bars.py \
  --baseline-root /ruta/a/baseline_setpoints_ae140bxydeg_north-east-west \
  --tarifa-json sinergym/data/tarifas/tarifas_ute.json \
  --building-epjson sinergym/data/buildings/idf_multiplesZonas_termostato_ae140bxydeg.epJSON \
  --output-dir /ruta/salida_eval_baselines
```

Con valores por defecto del script (solo válidos si `ROOT` coincide con tu máquina):

```bash
python scripts/eval_baselines_40_and_plot_bars.py
```

## Argumentos

| Argumento | Descripción |
|-----------|-------------|
| `--baseline-root` | Directorio que contiene subcarpetas `sp_XX` con `eplusout.csv` (patrón `sp_*/eplusout.csv`). |
| `--tarifa-json` | JSON de tarifas UTE (estructura: `precios`, `horarios`, días punta). |
| `--building-epjson` | Modelo **epJSON** usado solo para **detectar zonas activas**: revisa `Schedule:Compact` `ZonaEast`, `ZonaWest`, `ZonaNorth`, `ZonaSouth` y considera activa una zona si el schedule tiene algún valor numérico mayor que cero. |
| `--output-dir` | Carpeta donde escribir CSV y PNG. Si queda vacío tras parseo, se usa un nombre tipo `batch_eval_baselines_ae140bxydeg_<timestamp>`. |

## Entradas esperadas

1. **Un `eplusout.csv` por baseline**, ruta típica:

   `<baseline-root>/sp_23/eplusout.csv` → etiqueta `BL_sp23`.

2. El CSV debe incluir al menos:
   - Columna **`Date/Time`** en el formato que parsea el script (fecha `M/D` + hora).
   - Columna de **potencia bomba de calor** (se busca la primera existente entre nombres tipo `BOMBACALOR_HP:Heat Pump Electricity Rate [W](Hourly)` o `Heat Pump Electricity Rate [W](Hourly)`).
   - Por cada **zona activa** detectada en el epJSON, columnas de **temperatura y humedad relativa** horarias de `EAST|WEST|NORTH|SOUTH PERIMETER` (nombres largos EnergyPlus).

3. **`--building-epjson`**: si no hay ninguna zona “activa”, el script **falla** (no hay fallback).

## Métricas (resumen)

- **PMV:** regresión lineal `pmv_simple(T, RH)`; **desviación horaria** fuera de [-0,5, 0,5] vía `pmv_deviation`, sumada sobre zonas activas y horas.
- **Costo:** `(potencia_W / 1000) * precio_kWh` hora a hora según tarifa y ventana punta/fuera de punta.

## Salidas (en `--output-dir`)

| Archivo / carpeta | Contenido |
|-------------------|-----------|
| `resumen_baselines_40.csv` | Por baseline: `label`, ruta `source`, muestras, costo anual, PMV desviación acumulada, `status`, `message`. |
| `bar_costo_baselines_40.png` | Barras de costo anual. |
| `bar_pmv_baselines_40.png` | Barras de desviación PMV acumulada. |
| `bar_confort_pmv_baselines_40_stacked.png` | Barras apiladas con 4 franjas de confort PMV (aprox.). |
| `plots_pmv_may01_oct01_sin_exterior/` | PNG por baseline: PMV interior 1 may – 1 oct (sin serie de clima exterior); opcionalmente traza PMV con setpoint fijo si el `label` contiene `spNN`. |

## Relación con otros scripts

- El CSV `resumen_baselines_40.csv` puede consumirse desde **`eval_best_model_and_plot_bars.py`** (`--baseline-summary-csv`) para comparar modelos RL con estos baselines.
- La lógica de PMV/costo es **coherente** con `eval_best_model_and_plot_bars.py`, pero aquí la fuente es **`eplusout.csv`**, no `observations.csv` de un entorno Gym.
