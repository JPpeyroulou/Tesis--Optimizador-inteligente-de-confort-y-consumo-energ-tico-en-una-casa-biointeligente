"""
Cliente Home Assistant: leer actuadores (input_number), escribir sensores
y bandera de sincronizacion Simulador <-> Modelo AI.

Por defecto usa WebSocket (1 conexion). Si falla o use_websocket=False, usa REST.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Union

import requests
import websocket

logger = logging.getLogger(__name__)

HA_Session = Union["HAWebSocketClient", requests.Session]

ACTUATORS = [
    "input_number.zona_north",
    "input_number.zona_south",
    "input_number.zona_east",
    "input_number.zona_west",
    "input_number.temperatura_calefaccion",
]

SENSORS = [
    "sensor.north_heating_rate_sensor",
    "sensor.south_heating_rate_sensor",
    "sensor.east_heating_rate_sensor",
    "sensor.west_heating_rate_sensor",
    "sensor.outdoor_temperature_sensor",
    "sensor.outdoor_humidity_sensor",
    "sensor.north_air_temperature_sensor",
    "sensor.south_air_temperature_sensor",
    "sensor.east_air_temperature_sensor",
    "sensor.west_air_temperature_sensor",
    "sensor.north_air_humidity_sensor",
    "sensor.south_air_humidity_sensor",
    "sensor.east_air_humidity_sensor",
    "sensor.west_air_humidity_sensor",
    "sensor.heat_pump_power_sensor",
    "sensor.total_electricity_hvac_sensor",
]

SENSOR_UNITS: dict[str, str] = {
    "sensor.north_heating_rate_sensor": "W",
    "sensor.south_heating_rate_sensor": "W",
    "sensor.east_heating_rate_sensor": "W",
    "sensor.west_heating_rate_sensor": "W",
    "sensor.outdoor_temperature_sensor": "°C",
    "sensor.outdoor_humidity_sensor": "%",
    "sensor.north_air_temperature_sensor": "°C",
    "sensor.south_air_temperature_sensor": "°C",
    "sensor.east_air_temperature_sensor": "°C",
    "sensor.west_air_temperature_sensor": "°C",
    "sensor.north_air_humidity_sensor": "%",
    "sensor.south_air_humidity_sensor": "%",
    "sensor.east_air_humidity_sensor": "%",
    "sensor.west_air_humidity_sensor": "%",
    "sensor.heat_pump_power_sensor": "W",
    "sensor.total_electricity_hvac_sensor": "W",
}


def _http_to_ws(url: str) -> str:
    """Convierte URL HTTP a WebSocket: http://host:8123 -> ws://host:8123/api/websocket"""
    s = url.strip().rstrip("/")
    if s.startswith("https://"):
        base = "wss://" + s[8:]
    elif s.startswith("http://"):
        base = "ws://" + s[7:]
    else:
        base = "ws://" + s
    return f"{base}/api/websocket"


def default_actuators() -> dict[str, float]:
    """Actuadores en 0 (para el primer paso, sin leer de HA)."""
    return {e: 0.0 for e in ACTUATORS}


class HAWebSocketClient:
    """
    Cliente HA via WebSocket: 1 conexion para todas las operaciones.
    Evita WinError 10048 al no abrir multiples conexiones TCP.
    """

    def __init__(self, url: str, token: str) -> None:
        self.ws_url = _http_to_ws(url)
        self.token = token
        self._ws: websocket.WebSocket | None = None
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def connect(self, retries: int = 5, retry_delay: float = 3.0) -> None:
        """Conecta y autentica. Reintenta si hay WinError 10048 (puertos en TIME_WAIT)."""
        for attempt in range(retries):
            try:
                self._ws = websocket.WebSocket()
                self._ws.settimeout(30)
                self._ws.connect(self.ws_url)
                break
            except OSError as e:
                if "10048" in str(e) and attempt < retries - 1:
                    logger.warning("Conexion fallida (10048), reintento en %.0fs...", retry_delay)
                    time.sleep(retry_delay)
                else:
                    raise

        msg = json.loads(self._ws.recv())
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"HA WebSocket: esperado auth_required, recibido {msg}")

        self._ws.send(json.dumps({"type": "auth", "access_token": self.token}))
        msg = json.loads(self._ws.recv())
        if msg.get("type") == "auth_invalid":
            raise RuntimeError(f"HA auth invalido: {msg.get('message', msg)}")
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"HA WebSocket: esperado auth_ok, recibido {msg}")

    def close(self) -> None:
        """Cierra la conexion."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _send_and_recv(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Envia comando y espera el resultado con mismo id."""
        if not self._ws:
            raise RuntimeError("WebSocket no conectado")
        msg_id = self._next_id()
        payload["id"] = msg_id
        self._ws.send(json.dumps(payload))
        while True:
            raw = self._ws.recv()
            msg = json.loads(raw)
            if msg.get("id") == msg_id:
                return msg
            if msg.get("type") == "event":
                continue
            if msg.get("id") is not None and msg.get("id") != msg_id:
                continue

    def get_states(self) -> list[dict[str, Any]]:
        """Obtiene todos los estados de entidades."""
        r = self._send_and_recv({"type": "get_states"})
        if not r.get("success"):
            raise RuntimeError(f"get_states fallo: {r}")
        return r.get("result") or []

    def _entity_state(self, entity_id: str, states: list[dict[str, Any]] | None = None) -> str | None:
        """Estado de una entidad. Si states=None, hace get_states."""
        if states is None:
            states = self.get_states()
        for s in states:
            if s.get("entity_id") == entity_id:
                return s.get("state")
        return None

    def call_service(
        self,
        domain: str,
        service: str,
        service_data: dict[str, Any] | None = None,
        target: dict[str, Any] | None = None,
    ) -> None:
        """Llama un servicio de HA."""
        payload: dict[str, Any] = {
            "type": "call_service",
            "domain": domain,
            "service": service,
        }
        if service_data:
            payload["service_data"] = service_data
        if target:
            payload["target"] = target
        r = self._send_and_recv(payload)
        if not r.get("success"):
            raise RuntimeError(f"call_service fallo: {r}")


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _create_rest_session(token: str) -> requests.Session:
    """Crea Session REST con pool minimo para reducir conexiones."""
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def create_ha_session(
    url: str, token: str, use_websocket: bool = True
) -> HA_Session:
    """
    Crea cliente HA. Por defecto WebSocket (1 conexion).
    Si use_websocket=False o WebSocket falla, usa REST.
    """
    if use_websocket:
        try:
            client = HAWebSocketClient(url, token)
            client.connect()
            return client
        except OSError as e:
            if "10048" in str(e):
                logger.warning("WebSocket fallo (10048), usando REST como respaldo")
                sess = _create_rest_session(token)
                sess._ha_url = url  # type: ignore
                return sess
            raise
    sess = _create_rest_session(token)
    sess._ha_url = url  # type: ignore
    return sess


def fetch_actuators(
    url: str,
    token: str,
    verbose: bool = True,
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> dict[str, float]:
    """Obtiene los valores actuales de los actuadores desde Home Assistant."""
    out: dict[str, float] = {}
    client = session
    own = client is None
    if own:
        client = create_ha_session(url, token, use_websocket)
    try:
        if isinstance(client, HAWebSocketClient):
            states = client.get_states()
            for i, entity_id in enumerate(ACTUATORS, 1):
                state = client._entity_state(entity_id, states)
                if state not in (None, "unavailable", "unknown"):
                    try:
                        val = float(state)
                    except ValueError:
                        val = 0.0
                else:
                    val = 0.0
                out[entity_id] = val
                if verbose:
                    logger.info("  [%d/%d] LEIDO   %s = %s", i, len(ACTUATORS), entity_id, val)
        else:
            ha_url = getattr(client, "_ha_url", url)
            for i, entity_id in enumerate(ACTUATORS, 1):
                try:
                    r = client.get(_url(ha_url, f"/api/states/{entity_id}"), timeout=10)
                    r.raise_for_status()
                    state = r.json().get("state")
                    if state not in (None, "unavailable", "unknown"):
                        val = float(state)
                    else:
                        val = 0.0
                except Exception as e:
                    if verbose:
                        logger.warning("  [%d/%d] LEIDO   %s -> ERROR: %s", i, len(ACTUATORS), entity_id, e)
                    val = 0.0
                out[entity_id] = val
                if verbose:
                    logger.info("  [%d/%d] LEIDO   %s = %s", i, len(ACTUATORS), entity_id, val)
    finally:
        if own and hasattr(client, "close"):
            client.close()
    return out


def push_sensors(
    url: str,
    token: str,
    values: dict[str, float],
    verbose: bool = True,
    use_input_number: bool = True,
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> None:
    """Envia los valores de sensores a Home Assistant."""
    items = [(e, v) for e, v in values.items() if e in SENSORS]
    n = len(items)
    client = session
    own = client is None
    if own:
        client = create_ha_session(url, token, use_websocket)
    try:
        if isinstance(client, HAWebSocketClient):
            if use_input_number:
                for i, (entity_id, value) in enumerate(items, 1):
                    target = entity_id.replace("sensor.", "input_number.", 1)
                    unit = SENSOR_UNITS.get(entity_id, "")
                    try:
                        client.call_service(
                            "input_number",
                            "set_value",
                            service_data={"entity_id": target, "value": round(value, 4)},
                        )
                        logger.info("  [%d/%d] ESCRITO %s = %s %s", i, n, target, round(value, 4), unit)
                    except Exception as e:
                        logger.warning("  [%d/%d] ESCRITO %s -> ERROR: %s", i, n, target, e)
            else:
                for i, (entity_id, value) in enumerate(items, 1):
                    unit = SENSOR_UNITS.get(entity_id, "")
                    payload = {"state": str(round(value, 4)), "attributes": {"unit_of_measurement": unit}}
                    try:
                        r = requests.post(
                            _url(url, f"/api/states/{entity_id}"),
                            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                            json=payload,
                            timeout=10,
                        )
                        r.raise_for_status()
                        logger.info("  [%d/%d] ESCRITO %s = %s %s", i, n, entity_id, round(value, 4), unit)
                    except Exception as e:
                        logger.warning("  [%d/%d] ESCRITO %s -> ERROR: %s", i, n, entity_id, e)
        else:
            ha_url = getattr(client, "_ha_url", url)
            for i, (entity_id, value) in enumerate(items, 1):
                target = entity_id.replace("sensor.", "input_number.", 1) if use_input_number else entity_id
                unit = SENSOR_UNITS.get(entity_id, "")
                payload = {"state": str(round(value, 4)), "attributes": {"unit_of_measurement": unit}}
                try:
                    r = client.post(_url(ha_url, f"/api/states/{target}"), json=payload, timeout=10)
                    r.raise_for_status()
                    logger.info("  [%d/%d] ESCRITO %s = %s %s", i, n, target, round(value, 4), unit)
                except Exception as e:
                    logger.warning("  [%d/%d] ESCRITO %s -> ERROR: %s", i, n, target, e)
    finally:
        if own and hasattr(client, "close"):
            client.close()


def wait_for_sync_flag_off(
    url: str,
    token: str,
    entity_id: str,
    poll_interval: float = 1.0,
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> None:
    """Espera indefinidamente hasta que la bandera este OFF."""
    client = session
    own = client is None
    if own:
        client = create_ha_session(url, token, use_websocket)
    try:
        while True:
            try:
                if isinstance(client, HAWebSocketClient):
                    state = client._entity_state(entity_id)
                else:
                    ha_url = getattr(client, "_ha_url", url)
                    r = client.get(_url(ha_url, f"/api/states/{entity_id}"), timeout=10)
                    r.raise_for_status()
                    state = (r.json().get("state") or "").lower()
                if (state or "").lower() == "off":
                    return
            except Exception as e:
                logger.warning("Error leyendo bandera %s: %s", entity_id, e)
            time.sleep(poll_interval)
    finally:
        if own and hasattr(client, "close"):
            client.close()


SIMULATION_DATETIME_ENTITY = "input_datetime.simulation_datetime"

FORECAST_ENTITIES = [
    f"input_number.forecast_{i}_outdoor_temp"
    for i in range(6)
] + [
    f"input_number.forecast_{i}_outdoor_humidity"
    for i in range(6)
]


def set_simulation_datetime(
    url: str,
    token: str,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> None:
    """Actualiza la fecha/hora de simulacion en HA (input_datetime.simulation_datetime)."""
    dt_str = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
    client = session
    own = client is None
    if own:
        client = create_ha_session(url, token, use_websocket)
    try:
        if isinstance(client, HAWebSocketClient):
            client.call_service(
                "input_datetime",
                "set_datetime",
                service_data={"entity_id": SIMULATION_DATETIME_ENTITY, "datetime": dt_str},
            )
        else:
            ha_url = getattr(client, "_ha_url", url)
            r = client.post(
                _url(ha_url, "/api/services/input_datetime/set_datetime"),
                json={"entity_id": SIMULATION_DATETIME_ENTITY, "datetime": dt_str},
                timeout=10,
            )
            r.raise_for_status()
        logger.info("  Fecha/hora simulacion -> %s", dt_str)
    except Exception as e:
        logger.warning("Error actualizando simulation_datetime: %s", e)
    finally:
        if own and hasattr(client, "close"):
            client.close()


FORECAST_24H_ENTITY = "input_text.forecast_24h_csv"


def set_forecast_24h_csv(
    url: str,
    token: str,
    rows: list[tuple[float, float, float, float, float, float]],
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> None:
    """
    Escribe 24h de pronóstico en input_text.forecast_24h_csv.
    rows: 24 filas; cada fila (temp_air °C, rh %, wind_dir °, wind_speed m/s, dni W/m², dhi W/m²).
    
    Formato compacto (max 255 chars en HA): temp,rh|temp,rh|... (24 pares separados por |)
    TODO: los otros 4 valores (wind_dir, wind_speed, dni, dhi) no caben; resolver mañana.
    """
    parts = []
    for row in rows[:24]:
        if len(row) >= 2:
            parts.append(f"{round(row[0], 1)},{round(row[1], 1)}")
        else:
            parts.append("0,0")
    while len(parts) < 24:
        parts.append("0,0")
    csv_content = "|".join(parts)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        requests.post(
            _url(url, f"/api/states/{FORECAST_24H_ENTITY}"),
            headers=headers,
            json={"state": csv_content},
            timeout=10,
        ).raise_for_status()
        logger.info("  Forecast 24h CSV escrito (%s)", FORECAST_24H_ENTITY)
    except Exception as e:
        logger.warning("Error escribiendo forecast_24h_csv: %s", e)


def set_sync_flag_on(
    url: str,
    token: str,
    entity_id: str,
    session: HA_Session | None = None,
    use_websocket: bool = True,
) -> None:
    """Pone la bandera en ON."""
    client = session
    own = client is None
    if own:
        client = create_ha_session(url, token, use_websocket)
    try:
        try:
            if isinstance(client, HAWebSocketClient):
                client.call_service(
                    "input_boolean",
                    "turn_on",
                    target={"entity_id": entity_id},
                )
            else:
                ha_url = getattr(client, "_ha_url", url)
                r = client.post(_url(ha_url, f"/api/states/{entity_id}"), json={"state": "on"}, timeout=10)
                r.raise_for_status()
            logger.info("  Bandera %s -> ON", entity_id)
        except Exception as e:
            logger.warning("Error poniendo bandera ON (%s): %s", entity_id, e)
    finally:
        if own and hasattr(client, "close"):
            client.close()
