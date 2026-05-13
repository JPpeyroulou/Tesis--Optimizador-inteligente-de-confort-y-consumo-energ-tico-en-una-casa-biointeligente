# EnergyPlus–Home Assistant Nexus

Puente entre el simulador de edificios **EnergyPlus** y **Home Assistant (HA)**: simula un año
completo, y luego, hora por hora, envía los sensores simulados a HA y lee los actuadores que
escribe el modelo de control (AI / lógica externa). Pensado para entrenar o probar agentes
tipo *sinergym* contra una "casa virtual" expuesta en HA.

---

## 1. ¿Qué hace?

En cada ciclo:

1. **Simula un año entero** con EnergyPlus usando el IDF (`idf_multiplesZonas_*.epJSON`) y
   el archivo de clima (`.epw`). El IDF **no se modifica** nunca.
2. Lee el CSV de salida (`eplusout.csv`) y genera una lista de sensores por hora.
3. **Por cada hora (paso):**
   - Espera la bandera de sincronización en **OFF** (el modelo AI ya leyó y actualizó actuadores).
   - Lee los actuadores actuales desde HA (`input_number.*`).
   - Escribe los sensores de esa hora en HA (`sensor.*` por defecto).
   - Actualiza la fecha/hora simulada (`input_datetime.simulation_datetime`) y el pronóstico
     a 24 h (`input_text.forecast_24h_csv`).
   - Pone la bandera en **ON** (el AI puede leer y decidir).
4. Cuando termina el año, **repite** desde el paso 1 (modo continuo) hasta `Ctrl+C`, o sale
   si se pasó `--once`.

```
┌──────────────┐      sensores (hora N)     ┌──────────────────┐
│  EnergyPlus  │ ─────────────────────────▶ │  Home Assistant  │
│  (un año)    │                            │   sensor.*       │
│              │ ◀───────────────────────── │   input_number.* │
└──────────────┘     actuadores (hora N)    └────────┬─────────┘
                                                     │
                                       lee sensores  │  escribe actuadores
                                                     ▼
                                            ┌──────────────────┐
                                            │  Modelo AI /     │
                                            │  control externo │
                                            └──────────────────┘
```

La coordinación entre simulador y AI se hace con un `input_boolean` (la **bandera de
sincronización**), de modo que ninguno avanza hasta que el otro terminó su turno.

---

## 2. Requisitos

- **Python 3.10+**
- **EnergyPlus** instalado (probado con V25-1-0 en Windows). Necesitas la ruta al ejecutable.
- Un archivo de clima **`.epw`** (incluido: `URY_Montevideo.epw`).
- Un **Home Assistant** alcanzable por red, con:
  - Las 5 entidades `input_number.*` para los actuadores.
  - Las entidades `sensor.*` (o `input_number.*`) donde el simulador escribirá.
  - Un `input_boolean` para la bandera (por defecto `input_boolean.sinergym_simulator_ready`).
  - Un `input_datetime.simulation_datetime` y un `input_text.forecast_24h_csv` (opcionales,
     pero recomendados).
  - Un **Long-Lived Access Token** (Perfil → Seguridad → Crear token).

Las dependencias Python están en `requirements.txt`:

```
requests>=2.28.0
pyyaml>=6.0
websocket-client>=1.6.0
```

---

## 3. Estructura del proyecto

```
energyplusNexus/
├── main.py                          # Orquestador: simula año + bucle por hora
├── run_energyplus.py                # Lanza EnergyPlus.exe (no modifica el IDF)
├── parse_output.py                  # Parser de eplusout.csv -> dict sensor.* por hora
├── ha_client.py                     # Cliente HA (WebSocket con fallback REST)
├── config.yaml                      # Tu config local (NO se commitea)
├── config.example.yaml              # Plantilla para empezar
├── requirements.txt
├── idf_multiplesZonas*.epJSON       # Modelo de edificio (4 zonas: N/S/E/W)
├── URY_Montevideo.epw               # Clima
├── homeassistant_templates_corregidos.yaml  # Templates para HA (si usás input_number)
├── eplus_run/                       # Salidas de EnergyPlus + CSV de logs por paso
└── config/                          # Documentación detallada
    ├── LECTURA_ACTUADORES_Y_SYNC.md
    ├── MODELO_AI_PROTOCOLO.md
    ├── API_HA_FECHA_SCHEDULE_FORECAST.md
    └── WINERROR_10048_HA.md
```

---

## 4. Instalación

Desde la carpeta del proyecto:

```powershell
# (opcional) crear y activar venv
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## 5. Configuración

### 5.1. `config.yaml`

Copiá la plantilla y editala:

```powershell
copy config.example.yaml config.yaml   # Windows
# cp config.example.yaml config.yaml   # Linux / macOS
```

Ejemplo mínimo:

```yaml
home_assistant:
  url: "http://127.0.0.1:8123"
  token: "YOUR_LONG_LIVED_ACCESS_TOKEN"
  sync_flag_entity: "input_boolean.sinergym_simulator_ready"
  use_input_number_for_sensors: false   # false = sensor.* (sinergym). true = input_number.*
  use_websocket: true                   # WebSocket = 1 conexión (evita WinError 10048)

energyplus:
  executable: "C:\\EnergyPlusV25-1-0\\EnergyPlus.exe"
  idf_path: "idf_multiplesZonas_ae140bxydeg.epJSON"
  weather_path: "URY_Montevideo.epw"
  run_directory: "eplus_run"
  simulation_year: 2021
  start_month: 6   # mes desde el que se sube data a HA (antes se saltan design days)
  start_day: 1
```

Claves importantes:

| Clave | Significado |
|---|---|
| `home_assistant.url` | URL de HA. Si HA corre en Docker, usá `127.0.0.1` si el puerto está mapeado al host; si está en otra máquina, usá su IP. |
| `home_assistant.token` | Long-Lived Access Token de HA. **No lo commitees.** |
| `home_assistant.sync_flag_entity` | `input_boolean` para sincronizar con el AI. Si lo dejás vacío, el simulador no espera a nadie. |
| `home_assistant.use_input_number_for_sensors` | `false` (default sinergym): escribe en `sensor.*` vía REST. `true`: escribe en `input_number.*` vía servicio. |
| `home_assistant.use_websocket` | `true`: una sola conexión WebSocket (recomendado en Windows). Si falla, hace fallback automático a REST. |
| `energyplus.start_month/day` | Fecha desde la que se envía data a HA (los *design days* iniciales se descartan). |

### 5.2. Entidades en Home Assistant

**Actuadores (los lee el simulador, los escribe el modelo AI):**

- `input_number.zona_north`
- `input_number.zona_south`
- `input_number.zona_east`
- `input_number.zona_west`
- `input_number.temperatura_calefaccion`

**Sensores (los escribe el simulador):**

Heating rate (W) por zona, temperatura/humedad exterior, temperatura/humedad por zona,
potencia de la bomba de calor y electricidad total HVAC. Listado completo en
`ha_client.py` (`SENSORS`).

**Sincronización:**

- `input_boolean.sinergym_simulator_ready` (configurable).
- `input_datetime.simulation_datetime` (fecha simulada actual).
- `input_text.forecast_24h_csv` (pronóstico de 24 h, formato `temp,rh|temp,rh|...`).

Si usás `use_input_number_for_sensors: true` (en vez de `sensor.*`), aplicá los templates
de `homeassistant_templates_corregidos.yaml` en tu HA para tener `sensor.*` derivados.

---

## 6. Uso

### 6.1. Dry-run (verificación, sin ejecutar EP ni tocar HA)

Comprueba que existan el IDF, el `.epw`, EnergyPlus y que la conexión con HA funcione.

```powershell
python main.py --dry-run
```

### 6.2. Un solo año y salir

```powershell
python main.py --once
```

### 6.3. Ejecución continua (default)

Año tras año hasta `Ctrl+C`:

```powershell
python main.py
```

### 6.4. Otras opciones

```powershell
python main.py --config otra_config.yaml      # usar otro YAML
python main.py --once --max-steps 10          # solo 10 pasos (debug rápido)
python main.py --once --verbose               # log detallado por entidad
python main.py --once --no-sync               # ignorar la bandera (no esperar al AI)
python main.py --no-push                      # leer actuadores pero NO escribir sensores
```

---

## 7. Cómo funciona internamente

### 7.1. `main.py` — orquestador

1. Carga `config.yaml`.
2. Crea **una** sesión HA (`create_ha_session` → WebSocket o REST).
3. Por cada "año":
   - Llama a `run_energyplus.run(...)` → ejecuta `EnergyPlus.exe -w clima.epw -r idf.epJSON`
     en `eplus_run/` y genera `eplusout.csv`.
   - `extract_sensor_values_all_hours(...)` parsea el CSV y devuelve una lista
     `[(fecha_hora, {sensor.x: valor, ...}), ...]`, una entrada por hora.
   - Salta los pasos iniciales hasta `start_month/start_day` (design days).
   - Itera por cada paso `(dt_str, sensors)`:
     - Si `step > 0` y hay `sync_entity`: `wait_for_sync_flag_off(...)`.
     - Si `step == 0`: actuadores en 0; si no, `fetch_actuators(...)`.
     - Loguea fila en `eplus_run/sensors_output_by_step.csv` y arma el forecast de 24 h
       (las próximas 24 entradas del año simulado).
     - `push_sensors(...)` → escribe los sensores en HA.
     - `set_simulation_datetime(...)` → actualiza `input_datetime.simulation_datetime`.
     - `set_forecast_24h_csv(...)` → empaqueta `temp,rh|...` en `input_text.forecast_24h_csv`.
     - `set_sync_flag_on(...)` → señal al AI: "podés leer".

### 7.2. `run_energyplus.py`

Copia el IDF y el `.epw` al `run_directory`, ejecuta:

```
EnergyPlus.exe -w URY_Montevideo.epw -d eplus_run -r idf_multiplesZonas_*.epJSON
```

`-r` activa **ReadVarsESO**, que produce `eplusout.csv` además del `.eso`.

### 7.3. `parse_output.py`

Lee `eplusout.csv` línea por línea. Cada columna se decodifica con un regex tipo
`KEY:Variable Name [Unit](Hourly)` y se mapea según la tabla `EP_TO_HA` a un
`sensor.<algo>_sensor`. La función expuesta a `main.py` es
`extract_sensor_values_all_hours(...)`, que devuelve una fila por hora.

### 7.4. `ha_client.py`

Cliente Home Assistant con **dos transportes**:

- **WebSocket** (`HAWebSocketClient`): una sola conexión persistente, autenticada con el
  token, que ejecuta `get_states` y `call_service`. Es el modo recomendado en Windows
  porque evita agotar puertos efímeros (WinError 10048).
- **REST** (`requests.Session`): fallback automático si el WebSocket falla, con pool
  mínimo (`pool_connections=1`).

Funciones principales:

| Función | Qué hace |
|---|---|
| `create_ha_session(url, token, use_websocket)` | Devuelve una sesión reutilizable. |
| `fetch_actuators(...)` | Lee los 5 `input_number.*` actuadores. |
| `push_sensors(...)` | Escribe `sensor.*` (REST `POST /api/states/...`) o `input_number.*` (servicio). |
| `wait_for_sync_flag_off(...)` | Bloquea hasta que el `input_boolean` esté en `off`. |
| `set_sync_flag_on(...)` | Llama `input_boolean.turn_on`. |
| `set_simulation_datetime(...)` | Llama `input_datetime.set_datetime` con la hora simulada. |
| `set_forecast_24h_csv(...)` | Empaqueta 24 pares `temp,rh` en `input_text.forecast_24h_csv` (máx 255 chars de HA). |

### 7.5. Protocolo de sincronización

```
PASO 0 (primer paso del año):                     PASO N > 0:
  actuadores = 0 (no leer HA)                       1. esperar bandera OFF
  escribir sensores                                 2. leer actuadores de HA
  bandera = ON                                      3. escribir sensores
                                                    4. bandera = ON
```

Detalle completo en `config/LECTURA_ACTUADORES_Y_SYNC.md` y el contrato esperado del
modelo externo en `config/MODELO_AI_PROTOCOLO.md`.

---

## 8. Salidas y logs

Dentro de `eplus_run/` vas a encontrar (entre otros):

- `eplusout.csv` — salida horaria de EnergyPlus (la que se parsea).
- `eplusout.eso`, `eplus*.htm`, `eplus*.err` — salidas estándar de EnergyPlus.
- `sensors_output_by_step.csv` — un volcado por paso con todos los sensores **y**
  actuadores leídos en cada hora. Útil para debug y para alimentar análisis posteriores.
- `forecast_by_step.csv` — el forecast 24 h calculado en cada paso.

Logs en consola (formato `YYYY-MM-DD HH:MM:SS [INFO] modulo: mensaje`). Con `--verbose`
se imprime cada lectura/escritura por entidad.

---

## 9. Notas y troubleshooting

- **`WinError 10048`** (Windows, agotamiento de puertos): mantené `use_websocket: true`;
  si aún así falla, ejecutá desde una terminal real (PowerShell/CMD), no desde la
  consola integrada del IDE. Más detalles en `config/WINERROR_10048_HA.md`.
- **HA en Docker:** usá `http://127.0.0.1:8123` si el puerto está mapeado al host
  (`-p 8123:8123`); si HA corre en otra máquina, usá su IP.
- **Token vacío o inválido:** `main.py` avisa con un warning y las llamadas a HA fallarán
  con 401. Regenerá el token en HA y actualizá `config.yaml`.
- **`use_input_number_for_sensors: false`** (default): el simulador escribe directamente en
  `sensor.*` vía `POST /api/states/...`. Compatible con sinergym. Con `true`, escribe en
  `input_number.*` y necesitás los templates de `homeassistant_templates_corregidos.yaml`
  para tener los `sensor.*` derivados.
- **El IDF no se modifica.** Si necesitás cambiar el modelo, editá tu propio epJSON y
  actualizá `energyplus.idf_path` en `config.yaml`.
- **`config.yaml` está en `.gitignore`** porque contiene tu token. No lo subas.

---

## 10. Referencias rápidas

- Lectura de actuadores y sincronización: `config/LECTURA_ACTUADORES_Y_SYNC.md`
- Contrato para el modelo AI externo: `config/MODELO_AI_PROTOCOLO.md`
- API HA (fecha, schedule, forecast): `config/API_HA_FECHA_SCHEDULE_FORECAST.md`
- Mitigación de WinError 10048: `config/WINERROR_10048_HA.md`
