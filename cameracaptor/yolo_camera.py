"""Interfaz local de detección de personas con YOLO."""

from __future__ import annotations

import argparse
import queue
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.camera import ReconnectingCamera
from src.config import CameraConfig, configure_opencv_ffmpeg, read_credentials
from src.detector import DetectionResult, YoloDetectorWorker
from src.events import DetectionEvent, EventLogger, handle_detection


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecta personas en el substream RTSP")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--every", type=int, default=3, help="Inferir uno de cada N frames")
    parser.add_argument("--device", default="auto", help="auto, cpu o índice CUDA")
    parser.add_argument("--model", default="yolo11n.pt")
    return parser.parse_args()


def add_panel(frame: np.ndarray, connected: bool, video_fps: float,
              result: DetectionResult | None, device: str) -> np.ndarray:
    height = frame.shape[0]
    panel = np.full((height, 300, 3), 28, dtype=np.uint8)
    white, green, gray = (235, 235, 235), (70, 220, 110), (170, 170, 170)
    cv2.putText(panel, "YOLO PERSONAS", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, white, 2)
    people = len(result.detections) if result else 0
    inference = result.inference_ms if result else 0.0
    rows = (("Conexion", "ACTIVA" if connected else "RECONECTANDO"),
            ("Video FPS", f"{video_fps:.1f}"), ("Personas", str(people)),
            ("Inferencia", f"{inference:.0f} ms"), ("Dispositivo", device.upper()))
    y = 75
    for label, value in rows:
        cv2.putText(panel, f"{label}:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gray, 1)
        cv2.putText(panel, value, (145, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, green, 1)
        y += 31
    cv2.putText(panel, "Q  Salir", (16, height - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(panel, "S  Captura", (16, height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    return np.hstack((frame, panel))


def main() -> int:
    args = arguments()
    if not 0 < args.confidence <= 1 or args.imgsz <= 0 or args.every <= 0:
        print("Error: confidence, imgsz y every deben ser válidos.")
        return 2
    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        print(f"Error: {error}.")
        return 2
    configure_opencv_ffmpeg()
    config = CameraConfig()
    print("Cargando YOLO; la primera ejecución puede descargar el modelo...")
    try:
        detector = YoloDetectorWorker(args.model, args.confidence, args.imgsz,
                                      (0,), args.device).start()
    except Exception as error:
        print(f"Error al cargar YOLO: {error}")
        return 1
    camera = ReconnectingCamera(config.make_url(username, password)).start()
    logger = EventLogger()
    samples: deque[float] = deque(maxlen=90)
    sequence = -1
    submitted = 0
    last_result: DetectionResult | None = None
    last_person_event = 0.0
    print(f"YOLO inicializando en {detector.device}. Solo se detectará la clase persona.")
    try:
        while True:
            try:
                while True:
                    print(detector.updates.get_nowait())
            except queue.Empty:
                pass
            if detector.failed:
                break
            snapshot = camera.latest(sequence)
            if snapshot is None:
                if cv2.waitKeyEx(20) & 0xFF in (ord("q"), ord("Q")):
                    break
                time.sleep(0.01)
                continue
            sequence, frame = snapshot.sequence, snapshot.frame
            clean_frame = frame.copy()
            now = time.monotonic()
            samples.append(now)
            video_fps = ((len(samples) - 1) / (samples[-1] - samples[0])
                         if len(samples) > 1 else 0.0)
            if sequence % args.every == 0:
                detector.submit(sequence, frame)
                submitted += 1
            current = detector.latest()
            if current is not None:
                last_result = current
            if last_result:
                for detection in last_result.detections:
                    x1, y1, x2, y2 = detection.coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 70), 2)
                    cv2.putText(frame, f"persona {detection.confidence:.0%}",
                                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (50, 220, 70), 2)
                if last_result.detections and now - last_person_event >= 10.0:
                    best = max(last_result.detections, key=lambda item: item.confidence)
                    event = DetectionEvent.create(config.name, "persona",
                                                  best.confidence, best.coordinates)
                    logger.write(event)
                    handle_detection(event)
                    last_person_event = now
            cv2.imshow("CameraCaptor YOLO",
                       add_panel(frame, camera.connected, video_fps, last_result, detector.device))
            key = cv2.waitKeyEx(1)
            ascii_key = chr(key & 0xFF).lower() if key != -1 else ""
            if ascii_key == "q":
                break
            if ascii_key == "s":
                Path("captures").mkdir(exist_ok=True)
                path = Path("captures") / f"yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), clean_frame)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
    print(f"Interfaz cerrada. Inferencias enviadas: {submitted}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
