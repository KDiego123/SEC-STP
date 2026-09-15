"""Demo interactiva: RTSP, movimiento, capturas, eventos y TTS local."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.camera import ReconnectingCamera
from src.config import CameraConfig, configure_opencv_ffmpeg, read_credentials
from src.events import DetectionEvent, EventLogger, handle_detection
from src.tts import WindowsSpeaker


PANEL_WIDTH = 330
EVENT_COOLDOWN = 10.0
MIN_MOTION_AREA = 1400


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo local de CameraCaptor")
    parser.add_argument("--credentials-file", type=Path)
    return parser.parse_args()


def draw_panel(
    frame: np.ndarray,
    connected: bool,
    fps: float,
    motion_enabled: bool,
    tts_enabled: bool,
    last_event: str,
    notice: str,
) -> np.ndarray:
    height = frame.shape[0]
    panel = np.full((height, PANEL_WIDTH, 3), (28, 28, 28), dtype=np.uint8)
    green, white, gray = (70, 210, 100), (235, 235, 235), (170, 170, 170)
    cv2.putText(panel, "CAMERACAPTOR", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, white, 2)
    rows = [
        ("Conexion", "ACTIVA" if connected else "RECONECTANDO"),
        ("FPS", f"{fps:.1f}"),
        ("Movimiento", "ON" if motion_enabled else "OFF"),
        ("Voz local", "ON" if tts_enabled else "OFF"),
    ]
    y = 76
    for label, value in rows:
        cv2.putText(panel, f"{label}:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)
        cv2.putText(panel, value, (158, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)
        y += 30
    cv2.line(panel, (18, y), (312, y), (75, 75, 75), 1)
    y += 30
    cv2.putText(panel, "Ultimo evento", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
    y += 25
    cv2.putText(panel, last_event[:34], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, gray, 1)
    if notice:
        cv2.putText(panel, notice[:34], (18, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1)
    controls = ["Q  Salir", "S  Captura", "M  Movimiento", "T  Voz"]
    y = height - 105
    for control in controls:
        cv2.putText(panel, control, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, white, 1)
        y += 23
    return np.hstack((frame, panel))


def save_snapshot(frame: np.ndarray) -> Path:
    directory = Path("captures")
    directory.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = time.strftime("%Y%m%d_%H%M%S")
    path = directory / f"capture_{stamp}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise OSError("OpenCV no pudo guardar la captura")
    return path


def main() -> int:
    args = arguments()
    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        print(f"Error: {error}.")
        return 2

    configure_opencv_ffmpeg()
    config = CameraConfig()
    camera = ReconnectingCamera(config.make_url(username, password)).start()
    speaker = WindowsSpeaker()
    logger = EventLogger()
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=True)
    motion_enabled = False
    tts_enabled = False
    sequence = -1
    samples: deque[float] = deque(maxlen=90)
    last_event_at = 0.0
    last_event = "Ninguno"
    notice = "M: activar movimiento"
    notice_until = time.monotonic() + 4.0
    warmup_until = time.monotonic() + 3.0
    print("Demo iniciada. Controles: Q salir, S captura, M movimiento, T voz.")

    try:
        while True:
            snapshot = camera.latest(sequence)
            if snapshot is None:
                key = cv2.waitKeyEx(20)
                if key != -1 and chr(key & 0xFF).lower() == "q":
                    break
                time.sleep(0.01)
                continue
            sequence = snapshot.sequence
            frame = snapshot.frame
            clean_frame = frame.copy()
            now = time.monotonic()
            samples.append(now)
            fps = (len(samples) - 1) / (samples[-1] - samples[0]) if len(samples) > 1 else 0.0

            motion_boxes: list[tuple[int, int, int, int]] = []
            mask = subtractor.apply(frame)
            if motion_enabled and now >= warmup_until:
                _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_MOTION_AREA:
                        continue
                    x, y, width, height = cv2.boundingRect(contour)
                    motion_boxes.append((x, y, width, height))
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 190, 255), 2)

            if motion_boxes and now - last_event_at >= EVENT_COOLDOWN:
                x1 = min(box[0] for box in motion_boxes)
                y1 = min(box[1] for box in motion_boxes)
                x2 = max(box[0] + box[2] for box in motion_boxes)
                y2 = max(box[1] + box[3] for box in motion_boxes)
                event = DetectionEvent.create(config.name, "movimiento", 1.0, (x1, y1, x2, y2))
                logger.write(event)
                handle_detection(event)
                last_event_at = now
                last_event = time.strftime("Movimiento %H:%M:%S")
                if tts_enabled:
                    speaker.say("Movimiento detectado en la cámara principal")

            visible_notice = notice if now < notice_until else ""
            display = draw_panel(
                frame, camera.connected, fps, motion_enabled, tts_enabled, last_event, visible_notice
            )
            cv2.imshow("CameraCaptor Demo", display)
            key = cv2.waitKeyEx(1)
            ascii_key = chr(key & 0xFF).lower() if key != -1 else ""
            if ascii_key == "q":
                break
            if ascii_key == "m":
                motion_enabled = not motion_enabled
                warmup_until = now + 2.0
                notice = f"Movimiento {'ACTIVADO' if motion_enabled else 'DESACTIVADO'}"
                notice_until = now + 2.5
            elif ascii_key == "t":
                tts_enabled = not tts_enabled
                notice = f"Voz {'ACTIVADA' if tts_enabled else 'DESACTIVADA'}"
                notice_until = now + 2.5
                if tts_enabled:
                    speaker.say("Alertas de voz activadas")
            elif ascii_key == "s":
                try:
                    path = save_snapshot(clean_frame)
                    last_event = f"Captura: {path.name}"
                    notice = "Captura guardada"
                    notice_until = now + 2.5
                except OSError:
                    last_event = "Error al guardar captura"
            elif key in (2424832, 2490368, 2555904, 2621440):
                notice = "PTZ aún no configurado"
                notice_until = now + 2.5
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        speaker.close()
        cv2.destroyAllWindows()
    print("Demo cerrada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
