"""Panel local con video, TTS hacia la cámara y controles PTZ seguros."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

# Las bibliotecas de inferencia mantienen por defecto sus hilos ocupados tras
# cada prediccion. La espera pasiva conserva la frecuencia de YOLO sin consumir
# CPU mientras el detector espera el siguiente fotograma.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2
from PIL import Image, ImageTk

from src.api_server import ApiCommand, ApiServerError, PanelApiBridge, PanelApiServer
from src.camera import ReconnectingCamera
from src.camera_audio import CameraAudioMonitor
from src.camera_speaker import CameraSpeaker, MAX_TEXT_LENGTH
from src.config import CameraConfig, configure_opencv_ffmpeg, read_credentials
from src.detector import DetectionResult, YoloDetectorWorker
from src.discovery import DEFAULT_CAMERA_MAC, resolve_camera_host
from src.lights import CameraWeb, LightsWorker
from src.person_trigger import (
    ENTRY_ALERT,
    PersonAlertSequence,
    PersonAnnouncement,
    PersonArrivalTrigger,
    welcome_alert,
)
from src.ptz import OnvifPtz, PtzError, PtzWorker


PERSON_ALERT_TAG = "person-arrival"
API_TTS_TAG_PREFIX = "api-tts:"
DETECTION_INTERVAL = 0.1
RESULT_MAX_AGE = 2.0
ALERT_RETRY_DELAY = 2.0
FAST_PRESENCE_SECONDS = 0.6
STRONG_PERSON_CONFIDENCE = 0.70
MAIN_STREAM_PATH = "/stream1"
SUB_STREAM_PATH = "/stream2"

# En equipos con muchos hilos OpenCV puede gastar mas CPU e incluso tardar mas
# por la coordinacion interna. Ocho conserva la menor latencia medida en este
# equipo y deja recursos disponibles para la interfaz y la captura RTSP.
cv2.setNumThreads(min(8, os.cpu_count() or 1))


class ControlPanel:
    def __init__(self, root: tk.Tk, config: CameraConfig, camera: ReconnectingCamera,
                 username: str, password: str, idle_seconds: float = 12.0,
                 presence_seconds: float = 1.5, faces_dir: Path | None = None,
                 face_threshold: float = 0.50,
                 fast_presence_seconds: float = FAST_PRESENCE_SECONDS,
                 strong_person_confidence: float = STRONG_PERSON_CONFIDENCE,
                 api_host: str = "127.0.0.1", api_port: int = 8765,
                 api_token_file: Path | None = None,
                 api_enabled: bool = True) -> None:
        self.root, self.config, self.camera = root, config, camera
        mainstream_config = CameraConfig(
            name=config.name,
            host=config.host,
            port=config.port,
            path=MAIN_STREAM_PATH,
        )
        self.mainstream_url = mainstream_config.make_url(username, password)
        self.mainstream_camera: ReconnectingCamera | None = None
        self.api_bridge = PanelApiBridge()
        self.api: PanelApiServer | None = None
        self.speaker = CameraSpeaker(config.host, config.port, config.path)
        self.audio = CameraAudioMonitor(
            config.make_url(username, password),
            Path(__file__).with_name("models") / "vosk-model-small-es-0.42",
            commands_enabled=True,
        )
        self.voice_commands_muted_until = 0.0
        known_names = (
            [path.name for path in faces_dir.iterdir() if path.is_dir()]
            if faces_dir is not None else []
        )
        self.speaker.preload(
            [ENTRY_ALERT, *(welcome_alert(name) for name in sorted(known_names))]
        )
        self.lights = LightsWorker(CameraWeb(config.host, username, password))
        model_path = Path(__file__).with_name("yolo11n.pt")
        model_name = str(model_path) if model_path.exists() else "yolo11n.pt"
        try:
            self.detector: YoloDetectorWorker | None = YoloDetectorWorker(
                model_name,
                allowed_classes=(0,),
                face_gallery=str(faces_dir) if faces_dir else None,
                face_model_dir=str(Path(__file__).with_name("models")),
                face_threshold=face_threshold,
            ).start()
        except Exception:
            self.detector = None
        try:
            self.ptz: PtzWorker | None = PtzWorker(OnvifPtz(config.host))
        except PtzError:
            self.ptz = None
        self.detection_sequence = -1
        self.display_sequence = -1
        self.held_direction: str | None = None
        self.held_speed = 0.3
        self.move_generation = 0
        self.closing = False
        self.video_photo: ImageTk.PhotoImage | None = None
        self.display_stream = tk.StringVar(value="substream")
        self.detection_enabled = tk.BooleanVar(value=self.detector is not None)
        self.detection_status = tk.StringVar(value="Cargando YOLO nano..." if self.detector
                                             else "YOLO nano no disponible; revisa yolo11n.pt.")
        self.person_trigger = PersonArrivalTrigger(idle_seconds, presence_seconds)
        self.fast_presence_seconds = fast_presence_seconds
        self.strong_person_confidence = strong_person_confidence
        self.alert_sequence = PersonAlertSequence()
        self.alert_queue: deque[PersonAnnouncement] = deque()
        self.last_detection: DetectionResult | None = None
        self.last_detection_sequence = -1
        self.last_detection_submit = 0.0
        self.alert_in_flight: PersonAnnouncement | None = None
        self.alert_retry_count = 0
        self.alert_retry_at = 0.0
        self.last_person_present = False
        self.last_camera_connected = camera.connected
        self.last_api_snapshot_at = 0.0
        self.status = tk.StringVar(value="Conectando video...")
        self.lamp_status = tk.StringVar(value="Consultando lámparas...")
        self.night_status = tk.StringVar(value="Consultando visión...")
        self.connection = tk.StringVar(value="RTSP: conectando")
        self.audio_listening = tk.BooleanVar(value=False)
        self.voice_commands_enabled = tk.BooleanVar(value=True)
        self.audio_status = tk.StringVar(value="Preparando micrófono y comandos...")
        self.api_status = tk.StringVar(value="API: preparando...")
        self.speed = tk.DoubleVar(value=0.3)

        root.title("CameraCaptor - video, YOLO, voz, focos y PTZ")
        root.geometry("990x680")
        root.minsize(880, 580)
        root.protocol("WM_DELETE_WINDOW", self.close)

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Status.TLabel", padding=(8, 5))

        outer = ttk.Frame(root, padding=(12, 10, 12, 8))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="CameraCaptor", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.connection).pack(side="right", padx=(12, 0))
        ttk.Label(header, textvariable=self.api_status).pack(side="right", padx=(12, 0))

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        stream_row = ttk.Frame(left)
        stream_row.pack(fill="x", pady=(0, 6))
        ttk.Label(stream_row, text="Calidad de vista:").pack(side="left")
        for value, label in (("substream", "Substream"),
                             ("mainstream", "Mainstream HD")):
            ttk.Radiobutton(
                stream_row,
                text=label,
                value=value,
                variable=self.display_stream,
                command=self.switch_display_stream,
            ).pack(side="left", padx=(8, 0))
        ttk.Label(
            stream_row,
            text="YOLO siempre usa el substream",
        ).pack(side="right")
        self.video = ttk.Label(left, text="Esperando frames de la cámara...", anchor="center")
        self.video.pack(fill="both", expand=True)

        yolo_box = ttk.LabelFrame(left, text="Detección inteligente", padding=(8, 5))
        yolo_box.pack(fill="x", pady=(7, 0))
        self.detector_checkbox = ttk.Checkbutton(
            yolo_box,
            text=(f"YOLO personas · validar {fast_presence_seconds:g}–"
                  f"{presence_seconds:g} s · "
                  f"rearme {idle_seconds:g} s"),
            variable=self.detection_enabled, command=self.toggle_detection)
        self.detector_checkbox.pack(side="left", anchor="w")
        if self.detector is None:
            self.detector_checkbox.state(["disabled"])
        ttk.Label(yolo_box, textvariable=self.detection_status,
                  anchor="e").pack(side="right", fill="x", expand=True, padx=(10, 0))

        right = ttk.Frame(main, width=300)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)
        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)

        voice_tab = ttk.Frame(tabs, padding=12)
        camera_tab = ttk.Frame(tabs, padding=10)
        ptz_tab = ttk.Frame(tabs, padding=12)
        tabs.add(voice_tab, text="Voz")
        tabs.add(camera_tab, text="Iluminación")
        tabs.add(ptz_tab, text="PTZ")

        ttk.Label(voice_tab, text=f"Texto para la cámara · máximo {MAX_TEXT_LENGTH}").pack(anchor="w")
        self.text = tk.Text(voice_tab, height=6, wrap="word", font=("Segoe UI", 10))
        self.text.pack(fill="x", pady=(8, 10))
        self.text.insert("1.0", "Prueba de voz de Camera Captor.")
        self.speak_button = ttk.Button(voice_tab, text="Hablar por la cámara", command=self.speak)
        self.speak_button.pack(fill="x")
        ttk.Separator(voice_tab).pack(fill="x", pady=12)
        ttk.Label(voice_tab,
                  text="La voz se sintetiza en este PC y se envía por RTSP. "
                       "El aviso de personas comparte este canal.",
                  wraplength=255, justify="left").pack(anchor="w")
        ttk.Separator(voice_tab).pack(fill="x", pady=12)
        microphone_box = ttk.LabelFrame(
            voice_tab, text="Micrófono y comandos", padding=9
        )
        microphone_box.pack(fill="x")
        self.listen_checkbox = ttk.Checkbutton(
            microphone_box,
            text="Escuchar audio de la cámara",
            variable=self.audio_listening,
            command=self.toggle_audio_listening,
        )
        self.listen_checkbox.pack(anchor="w")
        self.commands_checkbox = ttk.Checkbutton(
            microphone_box,
            text="Órdenes: prende/apaga las luces",
            variable=self.voice_commands_enabled,
            command=self.toggle_voice_commands,
        )
        self.commands_checkbox.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            microphone_box, textvariable=self.audio_status,
            wraplength=235, justify="left",
        ).pack(anchor="w", pady=(7, 0))

        self.mode_buttons: dict[str, ttk.Button] = {}
        lamp_box = ttk.LabelFrame(camera_tab, text="Focos blancos", padding=10)
        lamp_box.pack(fill="x", pady=(0, 10))
        lamp_row = ttk.Frame(lamp_box)
        lamp_row.pack(fill="x")
        for command, label in (("lamp_on", "Encender"), ("lamp_off", "Apagar")):
            button = ttk.Button(lamp_row, text=label,
                                command=lambda c=command: self.control_modes(c))
            button.pack(side="left", fill="x", expand=True, padx=(0, 4))
            button.state(["disabled"])
            self.mode_buttons[command] = button
        button = ttk.Button(lamp_box, text="Modo original",
                            command=lambda: self.control_modes("lamp_auto"))
        button.pack(fill="x", pady=(4, 0))
        button.state(["disabled"])
        self.mode_buttons["lamp_auto"] = button
        ttk.Label(lamp_box, textvariable=self.lamp_status,
                  wraplength=245).pack(anchor="w", pady=(7, 0))

        night_box = ttk.LabelFrame(camera_tab, text="Visión nocturna", padding=10)
        night_box.pack(fill="x")
        night_row = ttk.Frame(night_box)
        night_row.pack(fill="x")
        for command, label in (("night_on", "Activar"), ("night_off", "Desactivar")):
            button = ttk.Button(night_row, text=label,
                                command=lambda c=command: self.control_modes(c))
            button.pack(side="left", fill="x", expand=True, padx=(0, 4))
            button.state(["disabled"])
            self.mode_buttons[command] = button
        button = ttk.Button(night_box, text="Modo automático",
                            command=lambda: self.control_modes("night_auto"))
        button.pack(fill="x", pady=(4, 0))
        button.state(["disabled"])
        self.mode_buttons["night_auto"] = button
        ttk.Label(night_box, textvariable=self.night_status,
                  wraplength=245).pack(anchor="w", pady=(7, 0))
        ttk.Label(night_box,
                  text="Los focos blancos solo encienden en noche; con IR la imagen es gris.",
                  wraplength=245).pack(anchor="w", pady=(5, 0))
        self.modes_retry = ttk.Button(night_box, text="Reintentar estado",
                                      command=lambda: self.control_modes("read"))
        self.modes_retry.pack(fill="x", pady=(5, 0))
        self.modes_retry.state(["disabled"])

        ttk.Label(ptz_tab, text="Movimiento manual",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(ptz_tab, text="Mantén pulsada una flecha para mover.").pack(anchor="w", pady=(3, 8))
        grid = ttk.Frame(ptz_tab)
        grid.pack(pady=(2, 10))
        for direction, label, row, column in (
            ("up", "↑", 0, 1), ("left", "←", 1, 0),
            ("right", "→", 1, 2), ("down", "↓", 2, 1),
        ):
            button = ttk.Button(grid, text=label, width=5)
            button.grid(row=row, column=column, padx=4, pady=4)
            button.bind("<ButtonPress-1>", lambda event, d=direction: self.start_move(d))
            button.bind("<ButtonRelease-1>", lambda event, d=direction: self.stop_move(d))
            button.bind("<Leave>", lambda event, d=direction: self.stop_move(d))
            if self.ptz is None:
                button.state(["disabled"])
        ttk.Button(grid, text="■", width=5, command=self.stop_move).grid(
            row=1, column=1, padx=4, pady=4)
        ttk.Separator(ptz_tab).pack(fill="x", pady=(0, 10))
        ttk.Label(ptz_tab, text="Velocidad").pack(anchor="w")
        tk.Scale(ptz_tab, from_=0.1, to=0.6, resolution=0.05, orient="horizontal",
                 variable=self.speed).pack(fill="x")
        ttk.Label(ptz_tab, text="También funcionan las flechas del teclado. "
                  "Al soltar, la cámara se detiene.", wraplength=255).pack(anchor="w", pady=(8, 0))
        if self.ptz is None:
            ttk.Label(ptz_tab, text="PTZ no disponible por ONVIF.").pack(anchor="w", pady=(10, 0))

        status_bar = ttk.Frame(outer)
        status_bar.pack(fill="x", pady=(7, 0))
        ttk.Separator(status_bar).pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status, style="Status.TLabel",
                  anchor="w").pack(fill="x")
        if self.ptz is None:
            self.status.set("Video/voz listos; ONVIF PTZ no respondió.")
        for direction, keysym in (("up", "Up"), ("down", "Down"),
                                  ("left", "Left"), ("right", "Right")):
            root.bind_all(f"<KeyPress-{keysym}>",
                          lambda event, d=direction: self.key_press(event, d))
            root.bind_all(f"<KeyRelease-{keysym}>",
                          lambda event, d=direction: self.stop_move(d))
        root.bind("<FocusOut>", lambda event: self.stop_move())
        if api_enabled:
            token_path = api_token_file or Path(__file__).with_name("api_token.txt")
            try:
                self.api = PanelApiServer(
                    self.api_bridge, api_host, api_port, token_path
                ).start()
                self.api_status.set(f"API: {self.api.display_url}")
            except (ApiServerError, OSError, ValueError) as error:
                self.api_status.set(f"API: {error}")
        else:
            self.api_status.set("API: desactivada")
        root.after(40, self.refresh_video)
        root.after(100, self.refresh_status)

    def key_press(self, event: tk.Event, direction: str) -> None:
        if event.widget is self.text:
            return
        self.start_move(direction)

    def switch_display_stream(self) -> None:
        if self.closing:
            return
        self.display_sequence = -1
        if self.display_stream.get() == "mainstream":
            if self.mainstream_camera is None:
                self.mainstream_camera = ReconnectingCamera(
                    self.mainstream_url
                ).start()
            self.video.configure(image="", text="Conectando mainstream HD...")
            self.status.set("Abriendo vista principal; YOLO continúa en substream.")
        else:
            camera = self.mainstream_camera
            self.mainstream_camera = None
            if camera is not None:
                threading.Thread(
                    target=camera.close,
                    name="mainstream-close",
                    daemon=True,
                ).start()
            self.video.configure(image="", text="Cambiando a substream...")
            self.status.set("Vista substream activa.")
        self.api_bridge.add_event(
            "display_stream_changed", {"stream": self.display_stream.get()}
        )

    def start_move(self, direction: str) -> None:
        if self.ptz is None or self.closing or self.held_direction == direction:
            return
        self.begin_move(direction, self.speed.get())

    def begin_move(self, direction: str, speed: float,
                   duration_ms: int | None = None) -> None:
        if self.ptz is None or self.closing:
            return
        if self.held_direction is not None:
            self.ptz.stop()
        self.held_direction = direction
        self.held_speed = speed
        self.move_generation += 1
        generation = self.move_generation
        self.ptz.move(direction, speed)
        self.root.after(220, lambda: self.renew_move(generation))
        if duration_ms is not None:
            self.root.after(
                duration_ms, lambda: self.stop_move_generation(generation)
            )

    def renew_move(self, generation: int) -> None:
        if self.closing or self.ptz is None or generation != self.move_generation:
            return
        if self.held_direction is not None:
            self.ptz.move(self.held_direction, self.held_speed)
            self.root.after(220, lambda: self.renew_move(generation))

    def stop_move_generation(self, generation: int) -> None:
        if generation == self.move_generation:
            self.stop_move()

    def stop_move(self, direction: str | None = None) -> None:
        if direction is not None and direction != self.held_direction:
            return
        if self.held_direction is not None:
            self.held_direction = None
            self.move_generation += 1
        if self.ptz is not None:
            self.ptz.stop()

    def speak(self) -> None:
        phrase = self.text.get("1.0", "end-1c").strip()
        if not phrase:
            self.status.set("Escribe una frase antes de hablar.")
            return
        if len(phrase) > MAX_TEXT_LENGTH:
            self.status.set(f"La frase supera {MAX_TEXT_LENGTH} caracteres.")
            return
        if not self.speaker.say(phrase):
            self.status.set("La cámara ya está hablando; espera a que termine.")
            return
        self.suppress_voice_commands()
        self.speak_button.state(["disabled"])
        self.status.set("Preparando audio para la cámara...")

    def suppress_voice_commands(self, seconds: float = 2.0) -> None:
        self.voice_commands_muted_until = max(
            self.voice_commands_muted_until, time.monotonic() + seconds
        )

    def toggle_audio_listening(self) -> None:
        enabled = self.audio_listening.get()
        self.audio.set_playback(enabled)
        self.audio_status.set(
            "Abriendo escucha del micrófono..." if enabled
            else "Escucha apagada; los comandos pueden seguir activos."
        )

    def toggle_voice_commands(self) -> None:
        enabled = self.voice_commands_enabled.get()
        self.audio.set_commands(enabled)
        self.audio_status.set(
            "Preparando reconocimiento offline..." if enabled
            else "Comandos de voz desactivados."
        )

    def execute_voice_command(self, command: str) -> None:
        labels = {
            "voice_lights_on": "Prende las luces",
            "voice_lights_off": "Apaga las luces",
        }
        label = labels.get(command)
        if label is None or not self.voice_commands_enabled.get():
            return
        if (self.speaker.busy
                or time.monotonic() < self.voice_commands_muted_until):
            self.audio_status.set(
                f"Orden «{label}» ignorada mientras la cámara hablaba."
            )
            return
        if not self.lights.operate(command):
            self.audio_status.set(
                f"Orden «{label}» detectada; control de luces ocupado."
            )
            return
        for button in (*self.mode_buttons.values(), self.modes_retry):
            button.state(["disabled"])
        self.audio_status.set(f"Orden reconocida: «{label}».")
        self.status.set(f"Ejecutando por voz: «{label}».")

    def control_modes(self, command: str) -> None:
        if self.lights.operate(command):
            for button in (*self.mode_buttons.values(), self.modes_retry):
                button.state(["disabled"])
            self.status.set("Actualizando focos o visión de la cámara...")

    def finish_api_command(self, command: ApiCommand, succeeded: bool,
                           message: str) -> None:
        self.api_bridge.add_event(
            "command_dispatched" if succeeded else "command_rejected",
            {
                "command_id": command.command_id,
                "action": command.action,
                "message": message,
            },
        )
        self.status.set(f"API: {message}")

    def execute_api_command(self, command: ApiCommand) -> None:
        action, payload = command.action, command.payload
        if action == "tts":
            text = str(payload.get("text", "")).strip()
            accepted = bool(text) and self.speaker.say(
                text, tag=API_TTS_TAG_PREFIX + command.command_id
            )
            if accepted:
                self.suppress_voice_commands()
                self.speak_button.state(["disabled"])
            self.finish_api_command(
                command, accepted,
                "TTS enviado a la cámara." if accepted
                else "el canal TTS está ocupado.",
            )
            return

        if action in ("lights_on", "lights_off", "night_mode"):
            light_command = {
                "lights_on": "voice_lights_on",
                "lights_off": "voice_lights_off",
                "night_mode": {
                    "auto": "night_auto", "day": "night_off", "night": "night_on",
                }.get(str(payload.get("mode"))),
            }[action]
            accepted = light_command is not None and self.lights.operate(light_command)
            if accepted:
                for button in (*self.mode_buttons.values(), self.modes_retry):
                    button.state(["disabled"])
            self.finish_api_command(
                command, accepted,
                "orden de iluminación enviada." if accepted
                else "el control de iluminación está ocupado.",
            )
            return

        if action == "ptz_move":
            if self.ptz is None:
                self.finish_api_command(command, False, "PTZ no disponible.")
                return
            self.begin_move(
                str(payload["direction"]), float(payload["speed"]),
                int(payload["duration_ms"]),
            )
            self.finish_api_command(command, True, "movimiento PTZ iniciado.")
            return

        if action == "ptz_stop":
            if self.ptz is None:
                self.finish_api_command(command, False, "PTZ no disponible.")
                return
            self.stop_move()
            self.finish_api_command(command, True, "movimiento PTZ detenido.")
            return

        if action == "detection":
            if self.detector is None:
                self.finish_api_command(command, False, "YOLO no disponible.")
                return
            self.detection_enabled.set(bool(payload.get("enabled")))
            self.toggle_detection()
            self.finish_api_command(command, True, "estado de YOLO actualizado.")
            return

        self.finish_api_command(command, False, "orden desconocida.")

    def process_api_commands(self) -> None:
        for _ in range(16):
            try:
                command = self.api_bridge.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self.execute_api_command(command)
            except (KeyError, TypeError, ValueError) as error:
                self.finish_api_command(
                    command, False, f"parámetros inválidos ({type(error).__name__})."
                )

    def update_api_status(self) -> None:
        result = self.last_detection
        detections = result.detections if result is not None else ()
        identities = sorted({item.identity for item in detections if item.identity})
        display_camera = (
            self.mainstream_camera
            if self.display_stream.get() == "mainstream"
            else self.camera
        )
        self.api_bridge.update_status(
            camera_connected=self.camera.connected,
            display_connected=(
                display_camera is not None and display_camera.connected
            ),
            display_stream=self.display_stream.get(),
            detection_available=self.detector is not None,
            detection_enabled=(
                self.detector is not None and self.detection_enabled.get()
            ),
            people=len(detections) if self.last_person_present else 0,
            identities=identities if self.last_person_present else [],
            inference_ms=(round(result.inference_ms, 1) if result is not None else None),
            speaker_busy=self.speaker.busy,
            lights_busy=self.lights.busy,
            lamp_mode=self.lights.lamp_mode,
            night_mode=self.lights.night_mode,
            ptz_available=self.ptz is not None,
            voice_commands_enabled=self.voice_commands_enabled.get(),
        )

    def toggle_detection(self) -> None:
        self.person_trigger.reset()
        self.alert_sequence.reset()
        self.alert_queue.clear()
        self.alert_in_flight = None
        self.alert_retry_count = 0
        self.alert_retry_at = 0.0
        self.last_person_present = False
        self.last_detection = None
        if self.detector is not None:
            latest = self.detector.latest()
            if latest is not None:
                self.last_detection_sequence = latest.sequence
        self.last_detection_submit = 0.0
        self.detection_status.set("YOLO activo; esperando inactividad observada..."
                                  if self.detection_enabled.get() else "YOLO pausado.")

    def process_detection(self, result: DetectionResult) -> None:
        self.last_detection = result
        self.last_detection_sequence = result.sequence
        people = len(result.detections)
        identities = sorted({item.identity for item in result.detections if item.identity})
        identity_status = f" · Rostro: {', '.join(identities)}" if identities else ""
        self.detection_status.set(
            f"Personas: {people} · inferencia {result.inference_ms:.0f} ms · "
            f"{self.detector.device}{identity_status}")
        present = people > 0
        strongest_confidence = max(
            (item.confidence for item in result.detections), default=0.0
        )
        required_presence = (
            self.fast_presence_seconds
            if strongest_confidence >= self.strong_person_confidence
            else self.person_trigger.confirmation_seconds
        )
        self.last_person_present = present
        arrival_detected = self.person_trigger.observe(
            present, result.completed_at,
            confirmation_seconds=required_presence,
        )
        if arrival_detected:
            self.alert_sequence.arm()
            self.alert_retry_count = 0
            self.alert_retry_at = 0.0
            self.api_bridge.add_event("person_detected", {
                "people": people,
                "confidence": round(strongest_confidence, 4),
                "identities": identities,
            })
        if not present and self.person_trigger.idle_elapsed(result.completed_at):
            self.alert_sequence.reset()
            self.alert_queue.clear()
            self.alert_retry_count = 0
            self.alert_retry_at = 0.0
        elif present:
            announcements = self.alert_sequence.announcements(identities)
            welcomes = [item for item in announcements if item.kind == "welcome"]
            for announcement in welcomes:
                self.api_bridge.add_event("identity_recognized", {
                    "text": announcement.text,
                    "identities": identities,
                })
            if welcomes:
                # La identidad conocida desplaza cualquier aviso genérico pendiente.
                self.alert_queue = deque(
                    item for item in self.alert_queue if item.kind != "entry"
                )
                if (self.alert_in_flight is not None
                        and self.alert_in_flight.kind == "entry"):
                    priority = welcomes.pop(0)
                    if self.speaker.replace(priority.text, tag=PERSON_ALERT_TAG):
                        self.alert_in_flight = priority
                        self.suppress_voice_commands()
                        self.status.set(
                            f"Aviso prioritario: «{priority.text}»."
                        )
                    else:
                        self.speaker.interrupt(PERSON_ALERT_TAG)
                        self.alert_in_flight = None
                        welcomes.insert(0, priority)
                    self.alert_retry_count = 0
                    self.alert_retry_at = 0.0
                self.alert_queue.extend(welcomes)
            else:
                self.alert_queue.extend(announcements)
        self.try_person_alert()

    def try_person_alert(self) -> None:
        result = self.last_detection
        if (not self.alert_queue or self.alert_in_flight is not None
                or not self.last_person_present
                or result is None or time.monotonic() < self.alert_retry_at
                or time.monotonic() - result.completed_at > RESULT_MAX_AGE
                or not self.detection_enabled.get() or not self.camera.connected):
            return
        announcement = self.alert_queue[0]
        if self.speaker.say(announcement.text, tag=PERSON_ALERT_TAG):
            self.alert_queue.popleft()
            self.alert_in_flight = announcement
            self.suppress_voice_commands()
            self.speak_button.state(["disabled"])
            self.status.set(f"Enviando aviso: «{announcement.text}».")

    def refresh_video(self) -> None:
        if self.closing:
            return
        connected = self.camera.connected
        if connected != self.last_camera_connected:
            self.api_bridge.add_event(
                "camera_connection", {"connected": connected}
            )
            self.person_trigger.reset()
            self.alert_sequence.reset()
            self.alert_queue.clear()
            self.last_person_present = False
            self.alert_in_flight = None
            self.alert_retry_count = 0
            self.last_detection = None
            if self.detector is not None:
                latest = self.detector.latest()
                if latest is not None:
                    self.last_detection_sequence = latest.sequence
            self.last_camera_connected = connected

        now = time.monotonic()
        detection_snapshot = self.camera.latest(self.detection_sequence)
        if detection_snapshot is not None:
            self.detection_sequence = detection_snapshot.sequence
            if (self.detector is not None and self.detection_enabled.get()
                    and connected and now - self.last_detection_submit >= DETECTION_INTERVAL):
                self.detector.submit(
                    detection_snapshot.sequence, detection_snapshot.frame
                )
                self.last_detection_submit = now

        if self.detector is not None and self.detection_enabled.get() and connected:
            result = self.detector.latest()
            if (result is not None and result.sequence != self.last_detection_sequence
                    and now - result.completed_at <= RESULT_MAX_AGE):
                self.process_detection(result)

        display_name = self.display_stream.get()
        display_snapshot = None
        if display_name == "mainstream":
            if self.mainstream_camera is not None:
                display_snapshot = self.mainstream_camera.latest(self.display_sequence)
        else:
            display_snapshot = detection_snapshot

        if display_snapshot is not None:
            self.display_sequence = display_snapshot.sequence
            frame = display_snapshot.frame
            if (display_name == "substream"
                    and self.detector is not None
                    and self.detection_enabled.get() and connected
                    and self.last_detection is not None
                    and now - self.last_detection.completed_at <= RESULT_MAX_AGE):
                for detection in self.last_detection.detections:
                    x1, y1, x2, y2 = detection.coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 70), 2)
                    label = (f"{detection.identity} {detection.identity_score:.0%}"
                             if detection.identity else
                             f"persona {detection.confidence:.0%}")
                    cv2.putText(frame, label,
                                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (50, 220, 70), 2)
                    if detection.face_coordinates is not None:
                        fx1, fy1, fx2, fy2 = detection.face_coordinates
                        cv2.rectangle(
                            frame, (fx1, fy1), (fx2, fy2), (230, 150, 40), 2
                        )
            if now - self.last_api_snapshot_at >= 0.5:
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if encoded_ok:
                    self.api_bridge.set_snapshot(encoded.tobytes())
                    self.last_api_snapshot_at = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((660, 440))
            self.video_photo = ImageTk.PhotoImage(image)
            self.video.configure(image=self.video_photo, text="")

        if display_name == "mainstream":
            display_connected = (
                self.mainstream_camera is not None
                and self.mainstream_camera.connected
            )
            self.connection.set(
                f"YOLO sub: {'conectado' if connected else 'reconectando'} · "
                f"Vista HD: {'conectada' if display_connected else 'conectando'}"
            )
        else:
            self.connection.set(
                "Substream: conectado" if connected else "Substream: reconectando"
            )
        self.root.after(40, self.refresh_video)

    def refresh_status(self) -> None:
        if self.closing:
            return
        self.process_api_commands()
        if self.api is not None:
            try:
                while True:
                    self.api_status.set(self.api.updates.get_nowait())
            except queue.Empty:
                pass
        self.audio.set_playback_muted(self.speaker.busy)
        if self.speaker.busy:
            self.suppress_voice_commands()
        try:
            while True:
                self.audio_status.set(self.audio.updates.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                self.execute_voice_command(self.audio.commands.get_nowait())
        except queue.Empty:
            pass
        if self.audio.commands_failed:
            self.voice_commands_enabled.set(False)
            self.commands_checkbox.state(["disabled"])
        try:
            while True:
                speech = self.speaker.results.get_nowait()
                if (speech.tag is not None
                        and speech.tag.startswith(API_TTS_TAG_PREFIX)):
                    self.api_bridge.add_event("tts_completed", {
                        "command_id": speech.tag.removeprefix(API_TTS_TAG_PREFIX),
                        "succeeded": speech.succeeded,
                        "text": speech.text,
                    })
                    continue
                if speech.tag != PERSON_ALERT_TAG:
                    continue
                announcement = self.alert_in_flight
                if announcement is None or speech.text != announcement.text:
                    # Resultado tardío de un aviso que fue sustituido.
                    continue
                self.alert_in_flight = None
                if (not speech.succeeded and self.last_person_present
                        and self.detection_enabled.get() and announcement is not None
                        and self.alert_retry_count < 1):
                    self.alert_retry_count += 1
                    self.alert_retry_at = time.monotonic() + ALERT_RETRY_DELAY
                    self.alert_queue.appendleft(announcement)
                    self.status.set("El aviso TTS falló; se reintentará una vez.")
                else:
                    self.alert_retry_count = 0
                    self.alert_retry_at = 0.0
        except queue.Empty:
            pass
        if self.detector is not None:
            try:
                while True:
                    message = self.detector.updates.get_nowait()
                    self.detection_status.set(message)
                    if self.detector.failed:
                        self.detection_enabled.set(False)
                        self.detector_checkbox.state(["disabled"])
                        self.person_trigger.reset()
                        self.alert_sequence.reset()
                        self.alert_queue.clear()
                        self.alert_in_flight = None
                    elif self.detector.ready:
                        self.person_trigger.reset()
                        self.alert_sequence.reset()
                        self.alert_queue.clear()
            except queue.Empty:
                pass
        try:
            while True:
                message, lamp, night = self.lights.updates.get_nowait()
                self.status.set(message)
                lamp_name = ({0: "IR: blancos apagados", 1: "blancos encendidos" if night == 2
                              else "blancos seleccionados (esperando noche)",
                              2: "mixta/automática"}.get(lamp, "desconocida"))
                night_name = {0: "automática", 1: "diurna", 2: "nocturna",
                              3: "horario"}.get(night, "desconocida")
                self.lamp_status.set(f"Lámpara: {lamp_name}.")
                self.night_status.set(f"Visión: {night_name}.")
                self.api_bridge.add_event("camera_mode", {
                    "lamp_mode": lamp,
                    "night_mode": night,
                    "message": message,
                })
        except queue.Empty:
            pass
        if not self.lights.busy:
            available = self.lights.lamp_mode is not None and self.lights.night_mode is not None
            for button in self.mode_buttons.values():
                button.state(["!disabled"] if available else ["disabled"])
            self.modes_retry.state(["disabled"] if available else ["!disabled"])
        for worker in (self.speaker, self.ptz):
            if worker is None:
                continue
            try:
                while True:
                    self.status.set(worker.updates.get_nowait())
            except queue.Empty:
                pass
        self.try_person_alert()
        self.update_api_status()
        self.speak_button.state(["disabled"] if self.speaker.busy
                                else ["!disabled"])
        self.root.after(100, self.refresh_status)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.status.set("Cerrando conexiones...")
        self.root.update_idletasks()
        self.stop_move()
        if self.api is not None:
            self.api.close()
        self.audio.close()
        if self.ptz is not None:
            self.ptz.close()
        self.lights.close()
        if self.detector is not None:
            self.detector.close()
        self.speaker.close()
        if self.mainstream_camera is not None:
            self.mainstream_camera.close()
        self.camera.close()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Panel YOLO, TTS y PTZ de CameraCaptor")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--host", default=CameraConfig.host,
                        help="IP preferida; ONVIF la corrige si cambió")
    parser.add_argument("--camera-mac", default=DEFAULT_CAMERA_MAC,
                        help="MAC usada para identificar la cámara por ONVIF")
    parser.add_argument("--faces-dir", type=Path,
                        help="Galería local: una subcarpeta con fotos por identidad")
    parser.add_argument("--face-threshold", type=float, default=0.50,
                        help="Similitud mínima para mostrar una identidad (0-1)")
    parser.add_argument("--idle-seconds", type=float, default=12.0,
                        help="Segundos sin personas antes de permitir otro aviso (10-15)")
    parser.add_argument("--presence-seconds", type=float, default=1.5,
                        help="Validación de detecciones dudosas (0.5-5)")
    parser.add_argument("--fast-presence-seconds", type=float,
                        default=FAST_PRESENCE_SECONDS,
                        help="Validación de detecciones claras (0.3-1.5)")
    parser.add_argument("--strong-person-confidence", type=float,
                        default=STRONG_PERSON_CONFIDENCE,
                        help="Confianza YOLO para usar validación rápida (0.5-0.95)")
    parser.add_argument("--api-host", default="127.0.0.1",
                        help="Interfaz de escucha de la API; por defecto solo local")
    parser.add_argument("--api-port", type=int, default=8765,
                        help="Puerto HTTP de la API")
    parser.add_argument("--api-token-file", type=Path,
                        help="Archivo local para el token Bearer de la API")
    parser.add_argument("--no-api", action="store_true",
                        help="Inicia el panel sin servidor API")
    args = parser.parse_args()
    if not 10.0 <= args.idle_seconds <= 15.0:
        print("Error: --idle-seconds debe estar entre 10 y 15.")
        return 2
    if not 0.5 <= args.presence_seconds <= 5.0:
        print("Error: --presence-seconds debe estar entre 0.5 y 5.")
        return 2
    if not 0.3 <= args.fast_presence_seconds <= 1.5:
        print("Error: --fast-presence-seconds debe estar entre 0.3 y 1.5.")
        return 2
    if args.fast_presence_seconds > args.presence_seconds:
        print("Error: la validación rápida no puede superar la validación normal.")
        return 2
    if not 0.50 <= args.strong_person_confidence <= 0.95:
        print("Error: --strong-person-confidence debe estar entre 0.50 y 0.95.")
        return 2
    if not 0.30 <= args.face_threshold <= 0.90:
        print("Error: --face-threshold debe estar entre 0.30 y 0.90.")
        return 2
    if not 1 <= args.api_port <= 65535:
        print("Error: --api-port debe estar entre 1 y 65535.")
        return 2
    if args.faces_dir is not None and not args.faces_dir.is_dir():
        print("Error: --faces-dir no existe o no es una carpeta.")
        return 2
    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        print(f"Error: {error}.")
        return 2
    configure_opencv_ffmpeg()
    host, discovery_source = resolve_camera_host(args.host, args.camera_mac)
    print(f"Cámara: {host} ({discovery_source}).")
    config = CameraConfig(host=host, path=SUB_STREAM_PATH)
    camera = ReconnectingCamera(config.make_url(username, password)).start()
    try:
        root = tk.Tk()
        ControlPanel(
            root, config, camera, username, password,
            idle_seconds=args.idle_seconds,
            presence_seconds=args.presence_seconds,
            faces_dir=args.faces_dir,
            face_threshold=args.face_threshold,
            fast_presence_seconds=args.fast_presence_seconds,
            strong_person_confidence=args.strong_person_confidence,
            api_host=args.api_host,
            api_port=args.api_port,
            api_token_file=args.api_token_file,
            api_enabled=not args.no_api,
        )
        root.mainloop()
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
