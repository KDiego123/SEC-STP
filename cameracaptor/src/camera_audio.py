"""Audio RTSP de la cámara: escucha local y órdenes de voz offline."""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path


AUDIO_RATE = 16_000
AUDIO_CHUNK_BYTES = 3_200  # 100 ms, mono, PCM s16le.
COMMAND_COOLDOWN_SECONDS = 4.0
MIN_COMMAND_CONFIDENCE = 0.62
VOICE_COMMANDS = {
    "prende las luces": "voice_lights_on",
    "apaga las luces": "voice_lights_off",
}
VOICE_GRAMMAR = (*VOICE_COMMANDS, "[unk]")


class CameraAudioError(RuntimeError):
    pass


def normalize_speech(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^a-zñ ]+", " ", without_accents).split())


def command_from_result(result: dict[str, object],
                        min_confidence: float = MIN_COMMAND_CONFIDENCE) -> str | None:
    text = normalize_speech(str(result.get("text", "")))
    command = VOICE_COMMANDS.get(text)
    if command is None:
        return None
    words = result.get("result")
    if isinstance(words, list):
        confidences = [
            float(word["conf"])
            for word in words
            if isinstance(word, dict) and isinstance(word.get("conf"), (int, float))
        ]
        if confidences and sum(confidences) / len(confidences) < min_confidence:
            return None
    return command


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError) as error:
        raise CameraAudioError("FFmpeg no está disponible") from error


class CameraAudioMonitor:
    """Mantiene una entrada RTSP solo mientras escucha o reconoce órdenes."""

    def __init__(self, url: str, model_dir: Path,
                 commands_enabled: bool = True) -> None:
        self._url = url
        self._model_dir = model_dir
        self._playback = threading.Event()
        self._commands_enabled = threading.Event()
        if commands_enabled:
            self._commands_enabled.set()
        self._playback_muted = threading.Event()
        self._stop = threading.Event()
        self._changed = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._model = None
        self._recognizer = None
        self._commands_failed = False
        self._last_command_at = 0.0
        self.updates: queue.Queue[str] = queue.Queue()
        self.commands: queue.Queue[str] = queue.Queue(maxsize=4)
        self._thread = threading.Thread(
            target=self._run, name="camera-audio", daemon=True
        )
        self._thread.start()

    @property
    def playback_enabled(self) -> bool:
        return self._playback.is_set()

    @property
    def commands_enabled(self) -> bool:
        return self._commands_enabled.is_set()

    @property
    def commands_failed(self) -> bool:
        return self._commands_failed

    def set_playback(self, enabled: bool) -> None:
        self._playback.set() if enabled else self._playback.clear()
        self._changed.set()

    def set_commands(self, enabled: bool) -> None:
        if enabled:
            self._commands_failed = False
            self._commands_enabled.set()
        else:
            self._commands_enabled.clear()
            self._recognizer = None
        self._changed.set()

    def set_playback_muted(self, muted: bool) -> None:
        self._playback_muted.set() if muted else self._playback_muted.clear()

    def _active(self) -> bool:
        return self._playback.is_set() or self._commands_enabled.is_set()

    def _ensure_recognizer(self):
        if not self._commands_enabled.is_set():
            self._recognizer = None
            return None
        if self._recognizer is not None:
            return self._recognizer
        if not self._model_dir.is_dir():
            raise CameraAudioError(
                f"falta el modelo de voz {self._model_dir.name}"
            )
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except (ImportError, OSError) as error:
            raise CameraAudioError("Vosk no está instalado") from error
        try:
            SetLogLevel(-1)
            if self._model is None:
                self.updates.put("Cargando reconocimiento de voz offline...")
                self._model = Model(str(self._model_dir))
            recognizer = KaldiRecognizer(
                self._model, AUDIO_RATE,
                json.dumps(VOICE_GRAMMAR, ensure_ascii=False),
            )
            recognizer.SetWords(True)
        except Exception as error:
            self._model = None
            raise CameraAudioError(
                "no se pudo cargar el modelo de voz offline"
            ) from error
        self._recognizer = recognizer
        self.updates.put("Comandos de voz listos: prende/apaga las luces.")
        return recognizer

    def _start_ffmpeg(self) -> subprocess.Popen[bytes]:
        startup = None
        if hasattr(subprocess, "STARTUPINFO"):
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
        command = [
            _ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32768", "-analyzeduration", "500000",
            "-i", self._url, "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                startupinfo=startup,
            )
        except OSError as error:
            raise CameraAudioError("no se pudo iniciar FFmpeg para el micrófono") from error
        with self._process_lock:
            self._process = process
        return process

    @staticmethod
    def _close_output(output) -> None:
        if output is not None:
            try:
                output.stop()
                output.close()
            except Exception:
                pass

    def _open_output(self):
        try:
            import sounddevice as sound

            return sound.RawOutputStream(
                samplerate=AUDIO_RATE,
                blocksize=AUDIO_RATE // 10,
                channels=1,
                dtype="int16",
                latency="low",
            )
        except Exception as error:
            raise CameraAudioError(
                "Windows no tiene una salida de audio disponible"
            ) from error

    def _handle_recognition(self, recognizer, chunk: bytes) -> None:
        if recognizer is None or not recognizer.AcceptWaveform(chunk):
            return
        try:
            result = json.loads(recognizer.Result())
        except (TypeError, ValueError):
            return
        command = command_from_result(result)
        now = time.monotonic()
        if command is None or now - self._last_command_at < COMMAND_COOLDOWN_SECONDS:
            return
        self._last_command_at = now
        try:
            self.commands.put_nowait(command)
        except queue.Full:
            pass

    def _capture(self) -> None:
        recognizer = self._ensure_recognizer()
        process = self._start_ffmpeg()
        output = None
        next_output_retry = 0.0
        self.updates.put("Micrófono RTSP conectado.")
        try:
            if process.stdout is None:
                raise CameraAudioError("FFmpeg no entregó el canal de audio")
            while not self._stop.is_set() and self._active():
                chunk = process.stdout.read(AUDIO_CHUNK_BYTES)
                if not chunk:
                    raise CameraAudioError("se perdió el audio RTSP")

                if self._commands_enabled.is_set():
                    if recognizer is None:
                        recognizer = self._ensure_recognizer()
                    self._handle_recognition(recognizer, chunk)
                else:
                    recognizer = None
                    self._recognizer = None

                if self._playback.is_set() and not self._playback_muted.is_set():
                    if output is None and time.monotonic() >= next_output_retry:
                        try:
                            output = self._open_output()
                            output.start()
                            self.updates.put("Escucha del micrófono activada.")
                        except CameraAudioError as error:
                            self.updates.put(f"Micrófono: {error}.")
                            next_output_retry = time.monotonic() + 5.0
                    if output is not None:
                        try:
                            output.write(chunk)
                        except Exception:
                            self._close_output(output)
                            output = None
                            next_output_retry = time.monotonic() + 5.0
                            self.updates.put("Micrófono: se perdió la salida de audio.")
                elif output is not None:
                    self._close_output(output)
                    output = None
        finally:
            self._close_output(output)
            with self._process_lock:
                if self._process is process:
                    self._process = None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._active():
                self._changed.wait(0.5)
                self._changed.clear()
                continue
            try:
                self._capture()
            except CameraAudioError as error:
                if self._commands_enabled.is_set() and (
                        "Vosk" in str(error) or "modelo de voz" in str(error)):
                    self._commands_failed = True
                    self._commands_enabled.clear()
                self.updates.put(f"Micrófono: {error}; reintentando...")
            if not self._stop.is_set() and self._active():
                self._changed.wait(2.0)
                self._changed.clear()

    def close(self) -> None:
        self._stop.set()
        self._changed.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self._thread.join(timeout=5)
