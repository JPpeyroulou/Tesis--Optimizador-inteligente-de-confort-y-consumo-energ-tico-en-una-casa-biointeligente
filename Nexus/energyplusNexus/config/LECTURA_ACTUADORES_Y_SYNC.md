# Lectura de actuadores y sincronización con bandera

Documenta el flujo de lectura de actuadores en Home Assistant y la espera por la bandera de sincronización Simulador ↔ Modelo AI.

---

## 1. Flujo general (por paso/hora)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PASO N (N > 0)                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Esperar bandera OFF (simulador bloqueado hasta que el AI esté listo)     │
│  2. Leer actuadores de HA (input_number.zona_*, input_number.temperatura_*)  │
│  3. Obtener sensores de esa hora (desde eplusout.csv ya generado)            │
│  4. Escribir sensores en HA (sensor.*)                                       │
│  5. Poner bandera ON (el AI puede leer sensores y actualizar actuadores)     │
│  6. Siguiente paso                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PASO 0 (primer paso del año)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. NO esperar bandera (se avanza inmediatamente)                            │
│  2. Actuadores en 0 (no leer de HA)                                          │
│  3. Obtener sensores de esa hora                                             │
│  4. Escribir sensores en HA                                                  │
│  5. Poner bandera ON                                                         │
│  6. Siguiente paso                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entidades involucradas

### Actuadores (input_number) — el modelo AI escribe aquí

| Entity ID                       | Descripción              |
|--------------------------------|--------------------------|
| `input_number.zona_north`      | Zona norte               |
| `input_number.zona_south`      | Zona sur                 |
| `input_number.zona_east`       | Zona este                |
| `input_number.zona_west`       | Zona oeste               |
| `input_number.temperatura_calefaccion` | Setpoint calefacción |

### Sensores (sensor.*) — el simulador escribe aquí

El proyecto escribe en `sensor.*` cuando `use_input_number_for_sensors: false` (compatible con sinergym):

- Temperaturas: `sensor.outdoor_temperature_sensor`, `sensor.north_air_temperature_sensor`, etc.
- Humedades: `sensor.outdoor_humidity_sensor`, `sensor.north_air_humidity_sensor`, etc.
- Heating rate: `sensor.north_heating_rate_sensor`, etc.
- Energía: `sensor.heat_pump_power_sensor`, `sensor.total_electricity_hvac_sensor`

### Bandera de sincronización (input_boolean)

| Entity ID                            | Uso                                                |
|-------------------------------------|----------------------------------------------------|
| `input_boolean.sinergym_simulator_ready` | ON = simulador subió datos; OFF = AI listo, simulador puede avanzar |

---

## 3. Lectura de actuadores (`fetch_actuators`)

**Archivo:** `ha_client.py`

- **Entidades leídas:** Las definidas en `ACTUATORS` (5 `input_number`).
- **Método:**
  - **WebSocket:** `get_states()` y `_entity_state(entity_id, states)`.
  - **REST:** `GET /api/states/{entity_id}` por actuador.
- **Valor por defecto:** Si el estado es `unavailable`, `unknown` o error → `0.0`.
- **Log:** `[i/5] LEIDO input_number.zona_north = 0.5` (cuando `verbose=True`).

**Primer paso:** No se llama a `fetch_actuators`; se usa `default_actuators()` (todos en 0).

---

## 4. Espera de bandera OFF (`wait_for_sync_flag_off`)

**Archivo:** `ha_client.py`

- **Comportamiento:** Espera indefinida hasta que la bandera esté `OFF`.
- **Polling:** Cada `poll_interval` segundos (default 1.0).
- **Método:**
  - **WebSocket:** `_entity_state(sync_entity)` con `get_states()`.
  - **REST:** `GET /api/states/{sync_entity}`.
- **Condición de salida:** `state.lower() == "off"`.
- **Errores:** Si falla la lectura, log de warning y sigue esperando (reintentos infinitos).

**Cuándo se llama:** Solo si `sync_entity` está configurado y `step > 0` (no en el primer paso).

---

## 5. Poner bandera ON (`set_sync_flag_on`)

**Archivo:** `ha_client.py`

- **Método:**
  - **WebSocket:** `call_service("input_boolean", "turn_on", target={"entity_id": ...})`.
  - **REST:** `POST /api/states/{entity_id}` con `{"state": "on"}`.
- **Cuándo se llama:** Tras escribir los sensores en HA (cuando hay `sync_entity`).

---

## 6. Configuración (config.yaml)

```yaml
home_assistant:
  url: "http://127.0.0.1:8123"
  token: "YOUR_LONG_LIVED_ACCESS_TOKEN"
  sync_flag_entity: "input_boolean.sinergym_simulator_ready"
  use_input_number_for_sensors: false
  use_websocket: true
```

| Clave                      | Descripción                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `sync_flag_entity`         | `input_boolean` para sincronizar. Vacío = no esperar (modo sin sync).       |
| `use_input_number_for_sensors` | `false` = escribir en `sensor.*`. `true` = escribir en `input_number.*`. |
| `use_websocket`            | `true` = WebSocket (1 conexión). Si falla, fallback a REST.                 |

---

## 7. Opciones de línea de comandos

| Opción      | Efecto                                              |
|------------|------------------------------------------------------|
| `--no-sync`| No usar bandera; avanza sin esperar.                 |
| `--verbose`| Log detallado de cada lectura/escritura por entidad. |
| `--max-steps N` | Limitar a N pasos (horas) por año.             |
| `--once`   | Un solo año y salir.                                 |

---

## 8. Ciclo Simulador ↔ AI

1. Simulador escribe sensores → pone bandera **ON**.
2. El modelo AI (u otro proceso) ve ON y lee sensores.
3. El modelo AI actualiza actuadores y pone la bandera **OFF**.
4. El simulador, al ver OFF, avanza al siguiente paso, lee actuadores, escribe sensores, vuelve a poner ON.
5. Se repite para cada hora.

**Documentación para el código del modelo AI:** `config/MODELO_AI_PROTOCOLO.md` — esperar bandera ON, leer sensores, ejecutar modelo, subir actuadores, bajar bandera.

---

## 9. Conexión a HA

- **WebSocket:** Una conexión persistente para lecturas, escrituras y bandera.
- **REST fallback:** Si WebSocket falla (p. ej. WinError 10048), se usa `requests.Session` con pool mínimo.
- Ver `config/WINERROR_10048_HA.md` para más detalles sobre el error 10048.
