"""
Ejecuta EnergyPlus con el IDF original (sin modificaciones).
Usa --readvars para generar CSV y poder parsear salidas.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run(
    executable: str,
    idf_path: str | Path,
    weather_path: str | Path,
    run_directory: str | Path,
    readvars: bool = True,
) -> Path:
    """
    Ejecuta EnergyPlus. No modifica el IDF.

    executable: ruta a EnergyPlus.exe
    idf_path: ruta a idf_multiplesZonas.epJSON (o .idf)
    weather_path: ruta al .epw
    run_directory: directorio de ejecución (se crea si no existe)
    readvars: si True, usa -r para generar CSV vía ReadVarsESO

    Devuelve el directorio de ejecución (contiene eplusout.eso, eplusout.csv si readvars, etc.)
    """
    idf_path = Path(idf_path).resolve()
    weather_path = Path(weather_path).resolve()
    run_dir = Path(run_directory).resolve()
    exe = Path(executable).resolve()

    if not idf_path.is_file():
        raise FileNotFoundError(f"IDF no encontrado: {idf_path}")
    if not weather_path.is_file():
        raise FileNotFoundError(f"Weather no encontrado: {weather_path}")
    if not exe.is_file():
        raise FileNotFoundError(f"EnergyPlus no encontrado: {exe}")

    run_dir.mkdir(parents=True, exist_ok=True)
    # Copiar IDF y weather al run dir para que EP los encuentre (algunas versiones lo requieren)
    run_idf = run_dir / idf_path.name
    run_weather = run_dir / weather_path.name
    if run_idf.resolve() != idf_path:
        shutil.copy2(idf_path, run_idf)
    if run_weather.resolve() != weather_path:
        shutil.copy2(weather_path, run_weather)

    cmd: list[str] = [
        str(exe),
        "-w", str(run_weather.name),
        "-d", str(run_dir),
        str(run_idf.name),
    ]
    if readvars:
        cmd.insert(-1, "-r")

    logger.info("Ejecutando EnergyPlus: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if proc.returncode != 0:
        err = (proc.stderr or "") + (proc.stdout or "")
        raise RuntimeError(f"EnergyPlus falló (code={proc.returncode}):\n{err}")

    logger.info("EnergyPlus finalizó correctamente.")
    return run_dir
