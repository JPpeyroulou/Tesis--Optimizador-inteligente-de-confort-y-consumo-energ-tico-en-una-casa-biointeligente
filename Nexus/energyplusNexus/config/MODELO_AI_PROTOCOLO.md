# Protocolo para el código del modelo AI

Guía para el proceso que debe: esperar bandera ON, leer sensores, ejecutar el modelo, subir actuadores y bajar la bandera.

---

## 1. Flujo del modelo AI (bucle por paso)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BUCLE DEL MODELO AI (repetir para cada hora de simulación)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Esperar hasta que la bandera esté ON (1)                                 │
│  2. Leer sensores de HA                                                      │
│  3. Ejecutar el modelo con esos valores                                      │
│  4. Subir las acciones (actuadores) a HA                                     │
│  5. Bajar la bandera (poner OFF)                                             │
│  6. Volver al paso 1                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Entidad de la bandera

| Entity ID                            | Estados    | Significado                                 |
|-------------------------------------|------------|---------------------------------------------|
| `input_boolean.sinergym_simulator_ready` | `on` / `off` | `on` = simulador subió sensores; `off` = AI terminó y subió actuadores |

- **ON (1):** El simulador (energyplusNexus) ya escribió los sensores. El modelo puede leer y procesar.
- **OFF (0):** El modelo ya actualizó actuadores. El simulador puede avanzar al siguiente paso.

---

## 3. Sensores que debes leer

El simulador escribe en estas entidades `sensor.*`:

| Entity ID                           | Unidad | Descripción              |
|-------------------------------------|--------|--------------------------|
| `sensor.outdoor_temperature_sensor` | °C     | Temperatura exterior     |
| `sensor.outdoor_humidity_sensor`    | %      | Humedad exterior         |
| `sensor.north_air_temperature_sensor` | °C   | Temp. zona norte         |
| `sensor.south_air_temperature_sensor` | °C   | Temp. zona sur           |
| `sensor.east_air_temperature_sensor`  | °C   | Temp. zona este          |
| `sensor.west_air_temperature_sensor`  | °C   | Temp. zona oeste         |
| `sensor.north_air_humidity_sensor`  | %      | Humedad zona norte       |
| `sensor.south_air_humidity_sensor`  | %      | Humedad zona sur         |
| `sensor.east_air_humidity_sensor`   | %      | Humedad zona este        |
| `sensor.west_air_humidity_sensor`   | %      | Humedad zona oeste       |
| `sensor.north_heating_rate_sensor`  | W      | Heating rate norte       |
| `sensor.south_heating_rate_sensor`  | W      | Heating rate sur         |
| `sensor.east_heating_rate_sensor`   | W      | Heating rate este        |
| `sensor.west_heating_rate_sensor`   | W      | Heating rate oeste       |
| `sensor.heat_pump_power_sensor`     | W      | Potencia bomba de calor  |
| `sensor.total_electricity_hvac_sensor` | W    | Electricidad HVAC total  |

---

## 4. Actuadores que debes escribir

| Entity ID                       | Descripción              |
|--------------------------------|--------------------------|
| `input_number.zona_north`      | Valor zona norte (0–1)   |
| `input_number.zona_south`      | Valor zona sur (0–1)     |
| `input_number.zona_east`       | Valor zona este (0–1)    |
| `input_number.zona_west`       | Valor zona oeste (0–1)   |
| `input_number.temperatura_calefaccion` | Setpoint calefacción (°C) |

---

## 5. API de Home Assistant

### Autenticación

Todas las peticiones llevan el header:

```
Authorization: Bearer YOUR_LONG_LIVED_ACCESS_TOKEN
Content-Type: application/json
```

### 5.1. Esperar bandera ON

Hacer polling hasta que la bandera sea `on`:

```
GET {url}/api/states/input_boolean.sinergym_simulator_ready
```

Respuesta (ejemplo):

```json
{
  "entity_id": "input_boolean.sinergym_simulator_ready",
  "state": "on",
  ...
}
```

- **Esperar ON:** Repetir el `GET` cada 1–2 segundos hasta que `state == "on"`.

### 5.2. Leer sensores

```
GET {url}/api/states/sensor.outdoor_temperature_sensor
GET {url}/api/states/sensor.north_air_temperature_sensor
...
```

O todos los estados:

```
GET {url}/api/states
```

Del JSON: `response["state"]` (string numérico) y `response["attributes"]["unit_of_measurement"]`.

### 5.3. Escribir actuadores

Usar el servicio `input_number.set_value`:

```
POST {url}/api/services/input_number/set_value
```

Cuerpo:

```json
{
  "entity_id": "input_number.zona_north",
  "value": 0.5
}
```

O actualizar el estado directamente:

```
POST {url}/api/states/input_number.zona_north
```

Cuerpo:

```json
{
  "state": "0.5",
  "attributes": {
    "unit_of_measurement": ""
  }
}
```

Hacer una llamada por actuador (o una llamada a `set_value` por actuador).

### 5.4. Bajar la bandera (OFF)

```
POST {url}/api/services/input_boolean/turn_off
```

Cuerpo:

```json
{
  "entity_id": "input_boolean.sinergym_simulator_ready"
}
```

O:

```
POST {url}/api/states/input_boolean.sinergym_simulator_ready
```

Cuerpo:

```json
{
  "state": "off"
}
```

---

## 6. Pseudocódigo

```python
import time
import requests

HA_URL = "http://127.0.0.1:8123"
TOKEN = "tu_long_lived_token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
BANDERA = "input_boolean.sinergym_simulator_ready"

SENSORES = [
    "sensor.outdoor_temperature_sensor",
    "sensor.north_air_temperature_sensor",
    # ... todos los de la tabla
]

ACTUADORES = [
    "input_number.zona_north",
    "input_number.zona_south",
    "input_number.zona_east",
    "input_number.zona_west",
    "input_number.temperatura_calefaccion",
]

def esperar_bandera_on():
    """Esperar hasta que la bandera esté ON."""
    while True:
        r = requests.get(f"{HA_URL}/api/states/{BANDERA}", headers=HEADERS)
        r.raise_for_status()
        if r.json().get("state", "").lower() == "on":
            return
        time.sleep(1)

def leer_sensores():
    """Leer todos los sensores y devolver dict entity_id -> valor float."""
    vals = {}
    for eid in SENSORES:
        r = requests.get(f"{HA_URL}/api/states/{eid}", headers=HEADERS)
        r.raise_for_status()
        try:
            vals[eid] = float(r.json().get("state", 0))
        except (ValueError, TypeError):
            vals[eid] = 0.0
    return vals

def escribir_actuadores(acciones):
    """acciones: dict entity_id -> valor float."""
    for eid, valor in acciones.items():
        requests.post(
            f"{HA_URL}/api/services/input_number/set_value",
            headers=HEADERS,
            json={"entity_id": eid, "value": valor},
        ).raise_for_status()

def bajar_bandera():
    """Poner la bandera en OFF."""
    requests.post(
        f"{HA_URL}/api/services/input_boolean/turn_off",
        headers=HEADERS,
        json={"entity_id": BANDERA},
    ).raise_for_status()

# Bucle principal
while True:
    esperar_bandera_on()
    sensores = leer_sensores()
    acciones = tu_modelo(sensores)  # tu_modelo(sensores) -> dict de actuadores
    escribir_actuadores(acciones)
    bajar_bandera()
```

---

## 7. Orden de ejecución (resumen)

| Orden | Acción                         | Cómo                           |
|-------|--------------------------------|--------------------------------|
| 1     | Esperar bandera ON             | `GET` a la bandera hasta `state == "on"` |
| 2     | Leer sensores                  | `GET /api/states/sensor.*`     |
| 3     | Ejecutar modelo                | Tu código con los valores      |
| 4     | Subir actuadores               | `POST` a `input_number/set_value` o `/api/states` |
| 5     | Bajar bandera                  | `input_boolean/turn_off`       |

---

## 8. Notas

- El simulador (energyplusNexus) debe estar en marcha con sync activado (`sync_flag_entity` definido y sin `--no-sync`).
- Usa una sola sesión HTTP (`requests.Session`) para evitar agotar puertos en Windows (ver `config/WINERROR_10048_HA.md`).
- La bandera puede estar ON desde el inicio (paso 0). El modelo debe estar listo para procesar el primer paso.
