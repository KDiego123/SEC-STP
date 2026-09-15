"""Encuentra una ruta RTSP válida sin mostrar ni almacenar credenciales."""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote


CANDIDATE_PATHS = (
    "/ch0_0.h264",
    "/ch0_1.h264",
    "/11",
    "/12",
    "/1",
    "/stream0",
    "/stream1",
    "/live/ch00_0",
    "/live/ch00_1",
    "/h264_stream",
    "/cam/realmonitor?channel=1&subtype=0",
    "/cam/realmonitor?channel=1&subtype=1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba rutas RTSP con ffprobe sin revelar credenciales."
    )
    parser.add_argument("--host", default="192.168.1.51", help="IP de la cámara")
    parser.add_argument("--port", type=int, default=554, help="Puerto RTSP")
    parser.add_argument(
        "--timeout", type=float, default=7.0, help="Segundos máximos por ruta"
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        help="Archivo local con 'usuario: valor' y 'contraseña: valor'",
    )
    return parser.parse_args()


def read_credentials(path: Path | None) -> tuple[str, str]:
    if path is None:
        return input("Usuario de la cámara: ").strip(), getpass.getpass(
            "Contraseña (oculta): "
        )

    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError) as error:
        raise ValueError("no se pudo leer el archivo de credenciales") from error
    values = []
    for line in lines:
        if line and ":" in line:
            values.append(line.split(":", 1)[1].strip())
    if len(values) != 2:
        raise ValueError("se esperaban dos líneas con formato 'etiqueta: valor'")
    return values[0], values[1]


def frame_rate(value: str | None) -> str:
    if not value or value == "0/0":
        return "desconocido"
    try:
        return f"{float(Fraction(value)):.2f}"
    except (ValueError, ZeroDivisionError):
        return "desconocido"


def probe(ffprobe: str, url: str, timeout: float) -> dict[str, object] | None:
    timeout_us = max(1, int(timeout * 1_000_000))
    command = [
        ffprobe,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        str(timeout_us),
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        url,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 2,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    # No se muestra stderr: algunos programas podrían repetir la URL con credenciales.
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    return streams[0] if streams else None


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535 or args.timeout <= 0:
        print("Error: el puerto y el tiempo de espera deben ser válidos.")
        return 2

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        print("No se encontró ffprobe.")
        print("Instala FFmpeg con: winget install --id Gyan.FFmpeg -e")
        print("Después cierra y abre la terminal y ejecuta: ffprobe -version")
        return 2

    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        print(f"Error: {error}.")
        return 2
    if not username or not password:
        print("Error: usuario y contraseña son obligatorios.")
        return 2

    encoded_user = quote(username, safe="")
    encoded_password = quote(password, safe="")
    authority = f"{encoded_user}:{encoded_password}@{args.host}:{args.port}"

    print(f"Probando {len(CANDIDATE_PATHS)} rutas en {args.host}:{args.port}...")
    for index, path in enumerate(CANDIDATE_PATHS, start=1):
        print(f"[{index}/{len(CANDIDATE_PATHS)}] Probando {path}")
        stream = probe(ffprobe, f"rtsp://{authority}{path}", args.timeout)
        if stream is None:
            continue
        codec = str(stream.get("codec_name") or "desconocido").upper()
        width = stream.get("width", "?")
        height = stream.get("height", "?")
        fps = frame_rate(str(stream.get("avg_frame_rate") or ""))
        print("Flujo RTSP válido encontrado:")
        print(f"  Ruta: {path}")
        print(f"  Códec: {codec}")
        print(f"  Resolución: {width}x{height}")
        print(f"  FPS declarados: {fps}")
        return 0

    print("No se encontró un flujo válido entre las rutas candidatas.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nPrueba cancelada.")
        raise SystemExit(130) from None
