# WinError 10048 al conectar con Home Assistant

## Causa

En Windows, el error **10048** ocurre cuando se agotan los puertos efímeros (muchas conexiones en TIME_WAIT).

## Solución implementada

1. **WebSocket (por defecto)**: 1 sola conexión para todas las operaciones.
2. **Fallback automático**: Si WebSocket falla con 10048, se usa REST con Session reutilizable.
3. **Config** `use_websocket: false`: forzar REST desde el inicio.

## Cómo está implementado

- **WebSocket**: `HAWebSocketClient` mantiene 1 conexión, usa `call_service` y `get_states`.
- **REST**: `requests.Session` con `pool_connections=1, pool_maxsize=1`.
- Si WebSocket falla con 10048, se usa REST automáticamente.

## Cómo comprobar

1. Ejecutar desde **tu terminal** (PowerShell/CMD), no desde el IDE:
   ```powershell
   cd c:\Users\JP\Desktop\folder\energyplusNexus
   python main.py --once --max-steps 3 --no-sync
   ```
2. Si sigue fallando, comprobar que Home Assistant está accesible:
   - HA en Docker: el puerto debe estar mapeado (`-p 8123:8123`).
   - HA en otra máquina: usar la IP en `config.yaml`: `url: "http://192.168.x.x:8123"`.
