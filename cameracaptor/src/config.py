"""Configuración y manejo local de credenciales."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass(frozen=True)
class CameraConfig:
    name: str = "camara-empresa-01"
    host: str = "192.168.1.51"
    port: int = 554
    path: str = "/ch0_0.h264"

    def make_url(self, username: str, password: str) -> str:
        user = quote(username, safe="")
        secret = quote(password, safe="")
        return f"rtsp://{user}:{secret}@{self.host}:{self.port}{self.path}"


def read_credentials(path: Path | None = None) -> tuple[str, str]:
    """Obtiene credenciales sin registrarlas ni mostrarlas."""
    if path is None:
        username = input("Usuario de la cámara: ").strip()
        password = getpass.getpass("Contraseña (oculta): ")
    else:
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError) as error:
            raise ValueError("no se pudo leer el archivo de credenciales") from error
        values = [line.split(":", 1)[1].strip() for line in lines if line and ":" in line]
        if len(values) != 2:
            raise ValueError("el archivo debe tener dos líneas con 'etiqueta: valor'")
        username, password = values
    if not username or not password:
        raise ValueError("usuario o contraseña vacíos")
    return username, password


def configure_opencv_ffmpeg() -> None:
    """Fuerza RTSP/TCP y limita esperas del backend FFmpeg de OpenCV."""
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000",
    )
