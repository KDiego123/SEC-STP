"""Panel local con video, TTS hacia la cámara y controles PTZ seguros."""

from __future__ import annotations

import argparse
import queue
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from src.camera import ReconnectingCamera
from src.camera_speaker import CameraSpeaker, MAX_TEXT_LENGTH
from src.config import CameraConfig, configure_opencv_ffmpeg, read_credentials
from src.detector import DetectionResult, YoloDetectorWorker
from src.lights import CameraWeb, LightsWorker
from src.person_trigger import PersonAlertSequence, PersonAnnouncement, PersonArrivalTrigger
from src.ptz import OnvifPtz, PtzError, PtzWorker


PERSON_ALERT_TAG = "person-arrival"
DETECTION_INTERVAL = 0.2
RESULT_MAX_AGE = 2.0
ALERT_RETRY_DELAY = 2.0


class ControlPanel:
    def __init__(self, root: tk.Tk, config: CameraConfig, camera: ReconnectingCamera,
                 username: str, password: str, idle_seconds: float = 12.0,
                 presence_seconds: float = 1.5, faces_dir: Path | None = None,
                 face_threshold: float = 0.50) -> None:
        self.root, self.config, self.camera = root, config, camera
        self.speaker = CameraSpeaker(config.host, config.port, config.path)
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
        self.sequence = -1
        self.held_direction: str | None = None
        self.move_generation = 0
        self.closing = False
        self.video_photo: ImageTk.PhotoImage | None = None
        self.detection_enabled = tk.BooleanVar(value=self.detector is not None)
        self.detection_status = tk.StringVar(value="Cargando YOLO nano..." if self.detector
                                             else "YOLO nano no disponible; revisa yolo11n.pt.")
        self.person_trigger = PersonArrivalTrigger(idle_seconds, presence_seconds)
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
        self.status = tk.StringVar(value="Conectando video...")
        self.lamp_status = tk.StringVar(value="Consultando lámparas...")
        self.night_status = tk.StringVar(value="Consultando visión...")
        self.connection = tk.StringVar(value="RTSP: conectando")
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

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.video = ttk.Label(left, text="Esperando frames de la cámara...", anchor="center")
        self.video.pack(fill="both", expand=True)

        yolo_box = ttk.LabelFrame(left, text="Detección inteligente", padding=(8, 5))
        yolo_box.pack(fill="x", pady=(7, 0))
        self.detector_checkbox = ttk.Checkbutton(
            yolo_box,
            text=(f"YOLO personas · validar {presence_seconds:g} s · "
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
        root.after(40, self.refresh_video)
        root.after(100, self.refresh_status)

    def key_press(self, event: tk.Event, direction: str) -> None:
        if event.widget is self.text:
            return
        self.start_move(direction)

    def start_move(self, direction: str) -> None:
        if self.ptz is None or self.closing or self.held_direction == direction:
            return
        if self.held_direction is not None:
            self.ptz.stop()
        self.held_direction = direction
        self.move_generation += 1
        generation = self.move_generation
        self.ptz.move(direction, self.speed.get())
        self.root.after(220, lambda: self.renew_move(generation))

    def renew_move(self, generation: int) -> None:
        if self.closing or self.ptz is None or generation != self.move_generation:
            return
        if self.held_direction is not None:
            self.ptz.move(self.held_direction, self.speed.get())
            self.root.after(220, lambda: self.renew_move(generation))

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
        self.speak_button.state(["disabled"])
        self.status.set("Preparando audio para la cámara...")

    def control_modes(self, command: str) -> None:
        if self.lights.operate(command):
            for button in (*self.mode_buttons.values(), self.modes_retry):
                button.state(["disabled"])
            self.status.set("Actualizando focos o visión de la cámara...")

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
        self.last_person_present = present
        if self.person_trigger.observe(present, result.completed_at):
            self.alert_sequence.arm()
            self.alert_retry_count = 0
            self.alert_retry_at = 0.0
        if not present and self.person_trigger.idle_elapsed(result.completed_at):
            self.alert_sequence.reset()
            self.alert_queue.clear()
            self.alert_retry_count = 0
            self.alert_retry_at = 0.0
        elif present:
            self.alert_queue.extend(self.alert_sequence.announcements(identities))
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
            self.speak_button.state(["disabled"])
            self.status.set(f"Enviando aviso: «{announcement.text}».")

    def refresh_video(self) -> None:
        if self.closing:
            return
        connected = self.camera.connected
        if connected != self.last_camera_connected:
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
        snapshot = self.camera.latest(self.sequence)
        if snapshot is not None:
            self.sequence = snapshot.sequence
            frame = snapshot.frame
            now = time.monotonic()
            if (self.detector is not None and self.detection_enabled.get()
                    and connected and now - self.last_detection_submit >= DETECTION_INTERVAL):
                self.detector.submit(snapshot.sequence, frame)
                self.last_detection_submit = now
            if self.detector is not None and self.detection_enabled.get() and connected:
                result = self.detector.latest()
                if (result is not None and result.sequence != self.last_detection_sequence
                        and now - result.completed_at <= RESULT_MAX_AGE):
                    self.process_detection(result)
                if (self.last_detection is not None
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
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (230, 150, 40), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((660, 440))
            self.video_photo = ImageTk.PhotoImage(image)
            self.video.configure(image=self.video_photo, text="")
        self.connection.set("RTSP: conectado" if connected else "RTSP: reconectando")
        self.root.after(40, self.refresh_video)

    def refresh_status(self) -> None:
        if self.closing:
            return
        try:
            while True:
                speech = self.speaker.results.get_nowait()
                if speech.tag != PERSON_ALERT_TAG:
                    continue
                announcement = self.alert_in_flight
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
        if self.ptz is not None:
            self.ptz.close()
        self.lights.close()
        if self.detector is not None:
            self.detector.close()
        self.speaker.close()
        self.camera.close()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Panel YOLO, TTS y PTZ de CameraCaptor")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--host", default=CameraConfig.host,
                        help="IP o nombre de red de la cámara")
    parser.add_argument("--faces-dir", type=Path,
                        help="Galería local: una subcarpeta con fotos por identidad")
    parser.add_argument("--face-threshold", type=float, default=0.50,
                        help="Similitud mínima para mostrar una identidad (0-1)")
    parser.add_argument("--idle-seconds", type=float, default=12.0,
                        help="Segundos sin personas antes de permitir otro aviso (10-15)")
    parser.add_argument("--presence-seconds", type=float, default=1.5,
                        help="Presencia continua antes del aviso (0.5-5)")
    args = parser.parse_args()
    if not 10.0 <= args.idle_seconds <= 15.0:
        print("Error: --idle-seconds debe estar entre 10 y 15.")
        return 2
    if not 0.5 <= args.presence_seconds <= 5.0:
        print("Error: --presence-seconds debe estar entre 0.5 y 5.")
        return 2
    if not 0.30 <= args.face_threshold <= 0.90:
        print("Error: --face-threshold debe estar entre 0.30 y 0.90.")
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
    config = CameraConfig(host=args.host, path="/stream2")
    camera = ReconnectingCamera(config.make_url(username, password)).start()
    try:
        root = tk.Tk()
        ControlPanel(
            root, config, camera, username, password,
            idle_seconds=args.idle_seconds,
            presence_seconds=args.presence_seconds,
            faces_dir=args.faces_dir,
            face_threshold=args.face_threshold,
        )
        root.mainloop()
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
