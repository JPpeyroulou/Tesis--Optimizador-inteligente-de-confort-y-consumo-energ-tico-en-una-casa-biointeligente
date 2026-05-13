# API Home Assistant: fecha/hora, schedule y forecast

Este documento explica cómo el **código externo** (por ejemplo el simulador EnergyPlus que escribe sensores y lee actuadores) debe actualizar vía API de Home Assistant:

1. **Fecha y hora de simulación** (para que el modelo RL use la hora de la lectura de los sensores).
2. **Forecast** (temperatura y humedad de las próximas horas).

Todas las entidades están en el dashboard de Sinergym (**Fecha, Schedule y Forecast**).

---

## Requisitos

- **URL de Home Assistant**: por ejemplo `http://localhost:8123` o `http://172.17.0.1:8123` desde Docker.
- **Token de acceso**: Long-Lived Access Token (Configuración → Usuarios → Tokens de acceso).
- Cabecera en todas las peticiones: `Authorization: Bearer <token>` y `Content-Type: application/json`.

---

## 1. Fecha y hora de simulación

El modelo RL toma **month**, **day_of_month** y **hour** desde esta entidad, no del reloj del sistema.

- **Entidad**: `input_datetime.simulation_datetime`
- **Formato del valor**: fecha y hora (HA lo muestra como `YYYY-MM-DD HH:MM:SS`).

### Cómo actualizarla (servicio)

Llamar al servicio `input_datetime.set_datetime` con la fecha y hora **de la lectura de los sensores** (por ejemplo la hora del timestep de EnergyPlus).

**Ejemplo con `curl`:**

```bash
curl -X POST "http://localhost:8123/api/services/input_datetime/set_datetime" \
  -H "Authorization: Bearer TU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "input_datetime.simulation_datetime",
    "datetime": "2025-02-03T14:30:00"
  }'
```

**Formato de `datetime`**:  
- Con hora: `"YYYY-MM-DDTHH:MM:SS"` (ej. `"2025-02-03T14:30:00"`).  
- Sin segundos: `"YYYY-MM-DDTHH:MM"` también suele ser aceptado.

**Ejemplo en Python (requests):**

```python
import requests

HA_URL = "http://localhost:8123"
TOKEN = "tu_long_lived_token"

def set_simulation_datetime(year: int, month: int, day: int, hour: int, minute: int = 0):
    url = f"{HA_URL}/api/services/input_datetime/set_datetime"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    dt = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
    requests.post(url, json={"entity_id": "input_datetime.simulation_datetime", "datetime": dt}, headers=headers)
```

Conviene actualizar esta entidad **cada vez que el código externo escribe los sensores** (por ejemplo después de avanzar un timestep en EnergyPlus), para que la fecha/hora coincida con la de los valores leídos por el modelo.

---

## 2. Forecast (24 horas: temp y humedad)

El modelo recibe **24 horas** de pronóstico con temperatura y humedad exterior.

- **Entidad**: `input_text.forecast_24h_csv`
- **Límite HA**: 255 caracteres (no se puede más en `input_text`).
- **Formato compacto**: 24 pares `temp,rh` separados por `|`:

```
21.7,86.1|21.3,88.5|20.9,90.3|20.6,92.1|20.4,93.3|...
```

Cada par: `temp_air` (°C), `rh` (%). El orden es hora +0, +1, ..., +23.

### Cómo leer el forecast (en el modelo RL)

```python
value = states("input_text.forecast_24h_csv")
parts = value.split('|')
for i, part in enumerate(parts):
    temp, rh = map(float, part.split(','))
    # hora +i: temp °C, rh %
```

### Cómo escribirlo (API)

Se usa `POST /api/states/input_text.forecast_24h_csv` con `{"state": "..."}`:

```python
def set_forecast_24h(temps: list, humidities: list):
    """temps y humidities: listas de 24 valores."""
    parts = [f"{round(temps[i],1)},{round(humidities[i],1)}" for i in range(24)]
    content = "|".join(parts)
    requests.post(
        f"{HA_URL}/api/states/input_text.forecast_24h_csv",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"state": content},
    )
```

El código externo debe actualizar el forecast **cuando actualice la fecha/hora y los sensores** (con los valores del .epw o del simulador para las próximas 24 horas).

**Nota:** Si necesitas los 6 valores originales (wind_dir, wind_speed, dni, dhi), considera usar un archivo local o múltiples entidades, ya que no caben en 255 caracteres.

---

## Orden recomendado en el código externo (EnergyPlus/simulador)

En cada timestep (o cada vez que se escriban los sensores):

1. Avanzar la simulación (EnergyPlus, etc.) y obtener:
   - Fecha/hora del timestep.
   - Valores de sensores (temperaturas, humedad, energía, etc.).
   - Pronóstico para las próximas 6 horas (temp/humedad).
2. **Escribir en HA** (en el orden que prefieras, antes de poner la bandera ON):
   - `input_datetime.simulation_datetime` → fecha y hora del timestep.
   - Sensores (según tu mapeo en `sensor_entities`).
   - `input_text.forecast_24h_csv` → CSV de 24 h × 6 valores (temp_air, rh, wind_dir, wind_speed, dni, dhi).
3. (Opcional) Actualizar `input_text.schedule_csv_content` si el schedule de presencia cambia.
4. Poner la bandera **ON** (`input_boolean.sinergym_simulator_ready`) para indicar que el modelo puede leer sensores y fecha/hora.

Así el modelo RL verá siempre la fecha/hora y el forecast alineados con la lectura de los sensores.

---

## Implementación en energyplusNexus

El simulador ya hace lo siguiente en cada paso (cuando no usa `--no-push`):

1. Escribe sensores (`sensor.*`).
2. Parsea la fecha/hora del CSV (`MM/DD  HH:MM:SS`) y actualiza `input_datetime.simulation_datetime` (año desde `energyplus.simulation_year` en config).
3. Escribe forecast 24 h en `input_text.forecast_24h_csv`: 24 filas con (temp_air, rh, wind_dir, wind_speed, dni, dhi). Por ahora temp_air y rh vienen del CSV de EnergyPlus; wind_dir, wind_speed, dni, dhi se envían a 0 (se pueden mapear luego si el IDF los genera).
4. Pone la bandera `input_boolean.sinergym_simulator_ready` en ON.

Configuración: `energyplus.simulation_year` (por defecto 2021) en `config.yaml`.
