# Entrenamiento con Sinergym

Esta guía describe cómo ejecutar un entrenamiento de aprendizaje por refuerzo **local** con el script del proyecto, qué archivos intervienen y qué debe estar instalado en el sistema.

## Requisitos previos

1. **Python** compatible con el `pyproject.toml` del repo (p. ej. Python 3.12).
2. **EnergyPlus** instalado y coherente con la versión soportada por Sinergym (véase `INSTALL.md`).
3. Paquete **sinergym** instalado en modo editable o desde el repo, con extra **DRL** para Stable Baselines 3 y callbacks:

   ```bash
   cd /ruta/al/clon/sinergym
   poetry install --extras drl
   # o, con pip:
   pip install -e ".[drl]"
   ```

4. Variable de entorno habitual de Sinergym/EnergyPlus si aplica en tu instalación (p. ej. ruta a EnergyPlus según la documentación oficial del simulador).

---

## Comando principal (entrenamiento local)

Desde la **raíz del repositorio** `sinergym`:

```bash
python scripts/train/local_confs/train_agent_local_conf.py \
  -conf scripts/train/local_confs/conf_examples/train_agent_SAC_nuestroMultizona.yaml
```

El único argumento obligatorio es **`-conf`**: ruta a un archivo YAML de experimento.

---

## Archivo obligatorio: YAML de experimento

El script lee **un solo YAML** que define todo el experimento. Ejemplos incluidos:

| Ruta (ejemplo) | Uso típico |
|----------------|------------|
| `scripts/train/local_confs/conf_examples/train_agent_PPO_nuestroMultizona.yaml` | PPO, multizona |
| `scripts/train/local_confs/conf_examples/train_agent_SAC_nuestroMultizona.yaml` | SAC, multizona |
| `scripts/train/local_confs/conf_examples/train_agent_PPO.yaml` | Plantilla genérica PPO |
| `scripts/train/local_confs/conf_examples/train_agent_SAC.yaml` | Plantilla genérica SAC |

### Claves que el script espera (resumen)

| Clave | Obligatoria | Descripción |
|-------|-------------|-------------|
| `experiment_name` | No | Prefijo del nombre de carpeta de salida. Si falta, se usa el nombre del algoritmo. |
| `environment` | **Sí** | ID Gymnasium del entorno (p. ej. `Eplus-nuestroMultizona-uru-continuous-v1`). Debe estar **registrado** al importar `sinergym`. |
| `episodes` | **Sí** | Número de episodios. Los pasos totales son `episodes × timestep_per_episode` del entorno. |
| `algorithm` | **Sí** | Incluye `name` (p. ej. `stable_baselines3:PPO`) y `parameters` para SB3. |
| `evaluation` | No | Si está definida, se crea entorno de evaluación y `LoggerEvalCallback` (`eval_freq`, `eval_length` en episodios). |
| `wrappers` | No | Lista de wrappers aplicados al entorno (p. ej. normalización, loggers). |
| `wrappers_yaml_config` | No | Ruta a YAML alternativo solo de wrappers. |
| `env_yaml_config` | No | Ruta a YAML con parámetros extra del entorno (sobrescribe/amplía la definición por defecto). |
| `env_params` | No | Diccionario en línea para sobrescribir parámetros del entorno (recompensa, semillas, etc.). |
| `env_deep_update` | No | Fusión profunda de dicts al mezclar parámetros (por defecto `true`). |
| `model` | No | Continuar desde un `.zip`: `local_path`, o artefacto W&B, o `bucket_path` (GCS). |
| `cloud` | No | Subida a bucket GCS y/o borrado de instancia MIG al terminar o ante error. |

Si activas **WandBLogger** en `wrappers`, necesitas API key y dependencias de `wandb`.

---

## Archivos que no pasas por línea de comandos pero deben existir

Al importar `sinergym`, los entornos `Eplus-*` se registran desde YAML en:

`sinergym/data/default_configuration/*.yaml` (p. ej. `nuestroMultrizona.yaml`).

Ese YAML de registro referencia, entre otros:

| Tipo | Ubicación típica en el paquete | Ejemplo (multizona) |
|------|--------------------------------|----------------------|
| Modelo del edificio | `sinergym/data/buildings/` | `idf_multiplesZonas_ae140bxydeg.epJSON` |
| Clima | `sinergym/data/weather/` | `URY_Montevideo.epw` |

Si cambias `building_file` o `weather_files` en la configuración del entorno, los ficheros correspondientes deben existir bajo las rutas que use Sinergym (normalmente dentro de `sinergym/data/`).

La **función de recompensa** y otros objetos se resuelven por rutas tipo `sinergym.utils.rewards:...` definidas en el YAML de registro del entorno.

---

## Salidas del entrenamiento

El script fija `env_name` al nombre del experimento con fecha. Suele generarse una carpeta de trabajo bajo el directorio actual del proceso, con:

- `progress.csv`, logs de episodios, **model** guardado al final (`model` en la ruta del workspace del entorno).
- Si hay evaluación: carpeta paralela con sufijo `_EVALUATION` y episodios numerados con CSVs en `monitor/`.

La ruta exacta depende de cómo `LoggerWrapper` / `CSVLogger` configuren el workspace (convención Sinergym).

---

## Otros flujos (referencia)

| Objetivo | Script / notas |
|----------|----------------|
| **Entrenamiento “producción” / online** | `scripts/train/prod/train_online_production.py --config <yaml_producción>` (Home Assistant, etc.); flujo distinto al local EnergyPlus. |

---

## Referencias

- Documentación oficial Sinergym: enlaces en `README.md` / `INSTALL.md`.
- Registro de entornos: comentario en `sinergym/data/default_configuration/*.yaml` y `sinergym/__init__.py` (`register_envs_from_yaml`).
