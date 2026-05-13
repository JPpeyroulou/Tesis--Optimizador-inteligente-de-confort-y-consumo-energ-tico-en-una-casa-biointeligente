# Evaluación de modelos y script `eval_best_model_and_plot_bars.py`

Esta guía separa dos ideas: la **evaluación durante el entrenamiento** (integrada en Sinergym/SB3) y la **evaluación por lotes** con el script de análisis de este repositorio.

---

## 1. Evaluación durante el entrenamiento

Cuando lanzas el entrenamiento con `scripts/train/local_confs/train_agent_local_conf.py` y el YAML incluye la sección `evaluation`, el script:

1. Crea un **segundo entorno** con nombre `\<experiment_name\>_EVALUATION`.
2. Registra un **`LoggerEvalCallback`** que ejecuta episodios de evaluación cada `eval_freq` episodios de entrenamiento, con política **determinista**.

**Archivos relevantes (por corrida de entrenamiento):**

| Ruta típica | Contenido |
|-------------|-----------|
| `.../evaluation/best_model.zip` | Mejor política según la lógica del callback de evaluación. |
| `.../evaluation/mean.txt`, `var.txt`, `count.txt` | Estadísticas de normalización de observaciones (si aplica) para alinear evaluación con el entrenamiento. |
| `.../evaluation/evaluation_metrics.csv` | Métricas agregadas de las evaluaciones. |
| `.../<nombre\>_EVALUATION-res*/episode-*/monitor/` | CSVs de observaciones, acciones, etc., de episodios de evaluación. |

Esa evaluación **no** es la que implementa `eval_best_model_and_plot_bars.py`; sirve para curvas de aprendizaje y para guardar `best_model.zip` dentro de la carpeta del entrenamiento.

---

## 2. Qué hace `eval_best_model_and_plot_bars.py`

Es un script **autónomo** que:

1. **Cataloga** un conjunto fijo de corridas de entrenamiento (PPO, SAC, TD3) mediante tablas internas de *timestamps* (`PPO_GROUPS`, `SAC_GROUPS`, `TD3_GROUPS`).
2. Para cada entrada, **localiza** la carpeta de entrenamiento con `glob` bajo `ROOT` (constante al inicio del script; por defecto apunta a `/workspaces/sinergym`).
3. Carga `evaluation/best_model.zip`, recrea el entorno Sinergym con los mismos wrappers base que el entrenamiento evaluable (`NormalizeAction`, `NormalizeObservation`, `LoggerWrapper`, `CSVLogger`).
4. Si existen `mean.txt` / `var.txt` / `count.txt` en `evaluation/`, **fija** la normalización de observaciones y **desactiva** la actualización online de media/varianza (para que la evaluación sea coherente con el modelo entrenado).
5. Ejecuta **un episodio completo** en bucle `reset` → `step` con `predict(..., deterministic=True)` hasta `terminated` o `truncated`.
6. Localiza el `observations.csv` generado en el workspace del logger.
7. Calcula **costo anual** (tarifa UTE desde JSON) y **desviación PMV acumulada** con una **fórmula aproximada** `pmv_simple(T, RH)` y penalización por salir de la banda ±0,5 (`pmv_deviation`).
8. Opcionalmente combina resultados con **baselines** leídos de un CSV de resumen (`--baseline-summary-csv`).
9. Genera **gráficas** (barras globales, PMV vs tiempo, tortas de confort, costos mensuales, subconjuntos modelo+baseline, etc.) y **CSVs** de resumen.

**Importante:** el script tiene `ROOT = "/workspaces/sinergym"` **hardcodeado**. Si tu clon está en otra ruta, debes editar esa constante o los valores por defecto de argumentos que concatenan `ROOT`, o invocar rutas absolutas en los flags.

---

## 3. Cómo ejecutarlo

Desde la raíz del repositorio (ajusta la ruta si tu layout difiere):

```bash
cd /workspaces/sinergym

python scripts/eval_best_model_and_plot_bars.py \
  --env-id Eplus-nuestroMultizona-uru-continuous-v1 \
  --tarifa-json sinergym/data/tarifas/tarifas_ute.json \
  --experiment-name Evaluacion-batch-best-model-barras \
  --building-epjson sinergym/data/buildings/idf_multiplesZonas_termostato_ae140bxydeg.epJSON \
  --baseline-summary-csv baseline_setpoints_ae140bxydeg_north-east-west/resumen_baselines_40.csv
```

Prueba rápida con solo los primeros modelos del catálogo interno:

```bash
python scripts/eval_best_model_and_plot_bars.py --max-models 2
```

**Solo regenerar gráficas** a partir de un CSV ya generado (no vuelve a simular):

```bash
python scripts/eval_best_model_and_plot_bars.py \
  --from /ruta/a/carpeta_batch \
  --output-dir /ruta/salida_opcional
```

En este modo debe existir `resumen_modelos_y_baselines.csv` dentro de la carpeta indicada.

---

## 4. Argumentos de línea de comandos

| Argumento | Por defecto (referencia) | Descripción |
|-----------|--------------------------|-------------|
| `--env-id` | `Eplus-nuestroMultizona-uru-continuous-v1` | ID del entorno Gymnasium registrado por Sinergym. |
| `--tarifa-json` | `ROOT/.../tarifas_ute.json` | Tarifa para costo eléctrico horario (estructura esperada: precios punta/fuera, horarios). |
| `--experiment-name` | `Evaluacion-batch-best-model-barras` | Prefijo del `env_name` al crear el entorno de evaluación. |
| `--output-dir` | vacío → `ROOT/batch_eval_<timestamp>` | Carpeta donde escribir CSVs y figuras. |
| `--max-models` | `0` | Si `>0`, solo procesa los primeros N modelos del catálogo interno (útil para depuración). |
| `--building-epjson` | ruta bajo `ROOT/.../idf_multiplesZonas_termostato_ae140bxydeg.epJSON` | Modelo usado para **detectar zonas activas** (schedules `ZonaEast`, etc.). |
| `--from` / `--from-existing-dir` | vacío | Si se define, modo solo gráficas desde `resumen_modelos_y_baselines.csv`. |
| `--baseline-summary-csv` | ruta bajo `ROOT/.../resumen_baselines_40.csv` | CSV con filas de baselines para tablas y figuras comparativas. |
| `--tabla-dd2-lotes` | opcional | Lista de IDs de lote (fecha/hora) para tabla d+d²; deben mapear a entradas en `*_GROUPS['d+d^2']`. |
| `--tabla-baseline-labels` | opcional | Etiquetas de baseline en el CSV (p. ej. `BL_sp23` `BL_sp24` `BL_sp25`). |

---

## 5. Archivos necesarios por modelo evaluado

Para cada combinación algoritmo + *timestamp* del catálogo interno:

| Archivo / condición | Rol |
|----------------------|-----|
| Carpeta que coincida con el patrón `Eplus-<ALG>-training-nuestroMultizona_*<timestamp>*-res*` | Directorio de una corrida de entrenamiento existente bajo `ROOT`. |
| `<run_dir>/evaluation/best_model.zip` | Política a cargar (obligatorio). |
| `<run_dir>/evaluation/mean.txt`, `var.txt`, (`count.txt`) | Normalización de observaciones al evaluar (muy recomendable que existan si el entrenamiento usó `NormalizeObservation`). |
| Entorno Sinergym + edificio + clima empaquetados | Resueltos por `create_environment` al importar `sinergym` (igual que en entrenamiento). |

Tras la simulación, el script exige encontrar al menos un **`observations.csv`** en el árbol del workspace del episodio.

---

## 6. Catálogo de corridas (`PPO_GROUPS`, `SAC_GROUPS`, `TD3_GROUPS`)

Al inicio del script hay diccionarios que agrupan **timestamps** de entrenamiento por algoritmo y por tipo de formulación de restricción en la recompensa (`d`, `d^2`, `d+d^2`). Cada elemento genera una etiqueta humana (`PPO_d+d2_r1`, etc.).

Para **añadir o cambiar** corridas evaluables, hay que editar esas listas y asegurarse de que en disco exista la carpeta de entrenamiento correspondiente.

---

## 7. Métricas de confort y costo (lógica propia del script)

- **`pmv_simple(tdb, rh)`:** regresión lineal corta en PMV (no es el PMV completo ISO/ASHRAE).
- **`pmv_deviation(pmv)`:** suma de “exceso” por encima de 0,5 y por debajo de −0,5 (penalización asimétrica por hora/zona según cómo se acumule en `compute_metrics_from_observations`).
- **Costo:** potencia de bomba de calor (columna esperada en `observations.csv`) × precio horario según `tarifas_ute.json` y calendario punta/fuera de punta.

Las **figuras de “confort”** (tortas, categorías PMV, etc.) combinan esta aproximación con reglas definidas en las funciones de agregación del propio script (ver código de `compute_pmv_category_percentages_*` y afines).

---

## 8. Salidas principales (directorio de salida)

Dependiendo de éxito parcial de cada bloque, se generan entre otros:

| Salida | Descripción aproximada |
|--------|-------------------------|
| `resumen_modelos_y_baselines.csv` | Tabla maestra de modelos evaluados (rutas, costo, PMV acum., estado). |
| `bar_*`, `scatter_*`, `pie_*` | Figuras PNG de barras, dispersión, tortas de confort. |
| `plots_pmv_background_algoritmos/` | Series temporales PMV interior vs exterior por zona. |
| Gráficos mayo–octubre | Ventana de temporada sin serie exterior en algunas figuras. |
| `tabla_costos_anual_y_mensual_dd2_y_baselines_sp23_sp24_sp25.csv` | Costo anual/mensual para subset d+d² + baselines. |
| Carpetas de **subset** y **bundle** completo | Comparativas modelo + baselines (ver llamadas a `save_selected_subset_bars` y `save_full_models_baselines_bundle` al final de `main`). |

Al terminar, el script imprime en consola un **resumen de rutas** de los artefactos generados.

---

## 9. Relación con `README_ENTRENAMIENTO.md`

- El **entrenamiento** produce las carpetas `Eplus-...-training-...-res*` y, si está configurado, `evaluation/best_model.zip`.
- Este script **consume** esos artefactos para una evaluación **batch** homogénea y informes de paper/informe.

Para problemas de rutas o entornos no encontrados, verifica `ROOT`, el `--env-id` y que los *timestamps* del catálogo coincidan con carpetas reales en disco.
