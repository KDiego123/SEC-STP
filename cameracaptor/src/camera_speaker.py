"""Texto de Windows SAPI enviado al altavoz mediante RTSP backchannel."""

from __future__ import annotations

import queue
import re
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin


BACKCHANNEL_TAG = "www.onvif.org/ver20/backchannel"
MAX_TEXT_LENGTH = 200
MAX_AUDIO_SECONDS = 20
SPEAKER_WAKE_AFTER_SECONDS = 45.0
SPEAKER_WAKE_AUDIO_SECONDS = 0.8
SPEAKER_WAKE_DELAY_SECONDS = 0.25


class SpeakerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechResult:
    text: str
    tag: str | None
    succeeded: bool


def _rtsp_response(connection: socket.socket) -> tuple[str, dict[str, str], bytes]:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 16384:
        chunk = connection.recv(4096)
        if not chunk:
            raise SpeakerError("la cámara cerró la conexión RTSP")
        data += chunk
    head, separator, body = data.partition(b"\r\n\r\n")
    if not separator:
        raise SpeakerError("respuesta RTSP incompleta")
    lines = head.decode("latin-1", errors="replace").split("\r\n")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.lower()] = value.strip()
    try:
        length = int(fields.get("content-length", "0"))
    except ValueError as error:
        raise SpeakerError("longitud RTSP inválida") from error
    if length > 100_000:
        raise SpeakerError("respuesta RTSP demasiado grande")
    while len(body) < length:
        chunk = connection.recv(min(4096, length - len(body)))
        if not chunk:
            raise SpeakerError("respuesta RTSP incompleta")
        body += chunk
    return lines[0], fields, body[:length]


def _request(
    connection: socket.socket,
    method: str,
    uri: str,
    sequence: int,
    extra: tuple[str, ...] = (),
) -> tuple[str, dict[str, str], bytes]:
    headers = (
        f"{method} {uri} RTSP/1.0",
        f"CSeq: {sequence}",
        "User-Agent: CameraCaptor-ControlPanel",
        *extra,
    )
    connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
    return _rtsp_response(connection)


def _backchannel_uri(sdp: str, content_base: str) -> str:
    current: dict[str, str] | None = None
    tracks: list[dict[str, str]] = []
    for line in sdp.splitlines():
        if line.startswith("m="):
            parts = line.split()
            current = {"type": parts[0][2:], "payload": parts[-1]}
            tracks.append(current)
        elif current is not None:
            if line.startswith("a=control:"):
                current["control"] = line.split(":", 1)[1]
            elif line.startswith("a=rtpmap:"):
                current["encoding"] = line.split(" ", 1)[-1]
            elif line in ("a=sendonly", "a=recvonly"):
                current["direction"] = line[2:]
    for track in tracks:
        if (
            track.get("type") == "audio"
            and track.get("direction") == "sendonly"
            and track.get("payload") == "0"
            and track.get("encoding") == "PCMU/8000"
            and track.get("control")
        ):
            return urljoin(content_base, track["control"])
    raise SpeakerError("la cámara no anunció un backchannel PCMU/8000")


def synthesize_pcmu(text: str) -> bytes:
    """Genera voz localmente y devuelve bytes G.711 μ-law a 8 kHz."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, OSError) as error:
            raise SpeakerError("no se encontró FFmpeg; reinstala requirements.txt") from error
    with tempfile.TemporaryDirectory(prefix="cameracaptor_speech_") as temp:
        wave_path = Path(temp) / "speech.wav"
        safe_path = str(wave_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$text=[Console]::In.ReadToEnd(); "
            "$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.SetOutputToWaveFile('{safe_path}'); "
            "$voice.Speak($text); $voice.Dispose()"
        )
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                input=text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
                startupinfo=startup,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise SpeakerError("Windows no pudo sintetizar la voz") from error
        if not wave_path.exists() or wave_path.stat().st_size < 1000:
            raise SpeakerError("Windows no produjo audio")
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(wave_path), "-ac", "1", "-ar", "8000",
                 "-f", "mulaw", "pipe:1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
                startupinfo=startup,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise SpeakerError("FFmpeg no pudo convertir el audio") from error
    audio = result.stdout
    if not audio or len(audio) > MAX_AUDIO_SECONDS * 8000:
        raise SpeakerError("audio vacío o superior a 20 segundos")
    return audio


def send_pcmu(
    host: str,
    port: int,
    path: str,
    audio: bytes,
    stop: threading.Event | None = None,
) -> int:
    """Abre una sesión temporal, envía audio y la cierra siempre."""
    base = f"rtsp://{host}:{port}{path}"
    require = f"Require: {BACKCHANNEL_TAG}"
    try:
        connection = socket.create_connection((host, port), timeout=5)
        connection.settimeout(5)
    except OSError as error:
        raise SpeakerError("no se pudo conectar con RTSP") from error
    session = ""
    packets = 0
    try:
        status, headers, body = _request(
            connection, "DESCRIBE", base, 1, ("Accept: application/sdp", require)
        )
        if not status.startswith("RTSP/1.0 200"):
            raise SpeakerError(f"la cámara rechazó DESCRIBE: {status}")
        content_base = headers.get("content-base", base + "/")
        back = _backchannel_uri(body.decode("latin-1", errors="replace"), content_base)
        status, headers, _ = _request(
            connection, "SETUP", back, 2,
            (require, "Transport: RTP/AVP/TCP;unicast;interleaved=0-1;mode=record"),
        )
        if not status.startswith("RTSP/1.0 200"):
            raise SpeakerError(f"la cámara rechazó SETUP: {status}")
        session = headers.get("session", "").split(";", 1)[0]
        if not session:
            raise SpeakerError("la cámara no devolvió sesión RTSP")
        channel_match = re.search(r"interleaved=(\d+)", headers.get("transport", ""))
        channel = int(channel_match.group(1)) if channel_match else 0
        status, _, _ = _request(
            connection, "PLAY", base, 3, (f"Session: {session}", require)
        )
        if not status.startswith("RTSP/1.0 200"):
            raise SpeakerError(f"la cámara rechazó PLAY: {status}")

        sequence = secrets.randbelow(65536)
        timestamp = secrets.randbits(32)
        ssrc = secrets.randbits(32)
        started = time.monotonic()
        for offset in range(0, len(audio), 160):
            if stop is not None and stop.is_set():
                break
            payload = audio[offset:offset + 160].ljust(160, b"\xff")
            rtp = struct.pack(
                "!BBHII", 0x80, 0x80 if offset == 0 else 0x00, sequence, timestamp, ssrc
            ) + payload
            connection.sendall(b"$" + bytes((channel,)) + struct.pack("!H", len(rtp)) + rtp)
            packets += 1
            sequence = (sequence + 1) & 0xFFFF
            timestamp = (timestamp + 160) & 0xFFFFFFFF
            delay = started + packets * 0.02 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        time.sleep(0.2)
        return packets
    except OSError as error:
        raise SpeakerError("se perdió la conexión durante el envío de audio") from error
    finally:
        if session:
            try:
                _request(connection, "TEARDOWN", base, 4, (f"Session: {session}", require))
            except (OSError, SpeakerError):
                pass
        connection.close()


class CameraSpeaker:
    """Trabajador de voz; nunca bloquea el hilo de interfaz."""

    def __init__(self, host: str, port: int, path: str) -> None:
        self.host, self.port, self.path = host, port, path
        self._messages: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._last_playback_at: float | None = None
        self.updates: queue.Queue[str] = queue.Queue()
        self.results: queue.Queue[SpeechResult] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="camera-speaker", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def say(self, text: str, tag: str | None = None) -> bool:
        text = text.strip().replace("\x00", "")
        if not text or len(text) > MAX_TEXT_LENGTH or self._busy.is_set():
            return False
        self._busy.set()
        try:
            self._messages.put_nowait((text, tag))
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def _wake_if_idle(self) -> None:
        now = time.monotonic()
        if (self._last_playback_at is not None
                and now - self._last_playback_at < SPEAKER_WAKE_AFTER_SECONDS):
            return
        self.updates.put("Activando el canal de audio de la cámara...")
        wake_audio = b"\xff" * int(8000 * SPEAKER_WAKE_AUDIO_SECONDS)
        try:
            send_pcmu(self.host, self.port, self.path, wake_audio, self._stop)
        except SpeakerError:
            # La sesión real se intenta igualmente: el precalentamiento es preventivo.
            pass
        self._stop.wait(SPEAKER_WAKE_DELAY_SECONDS)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text, tag = self._messages.get(timeout=0.2)
            except queue.Empty:
                continue
            succeeded = False
            try:
                self.updates.put("Sintetizando voz local...")
                audio = synthesize_pcmu(text)
                self._wake_if_idle()
                self.updates.put("Enviando audio a la cámara...")
                packets = send_pcmu(self.host, self.port, self.path, audio, self._stop)
                self._last_playback_at = time.monotonic()
                succeeded = True
                self.updates.put(f"Audio enviado ({packets * 0.02:.1f} s)")
            except SpeakerError as error:
                self.updates.put(f"Audio: {error}")
            finally:
                self.results.put(SpeechResult(text, tag, succeeded))
                self._busy.clear()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6)
