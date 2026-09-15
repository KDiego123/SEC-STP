"""Vista previa y prueba de estabilidad del substream RTSP."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from src.camera import ReconnectingCamera
from src.config import CameraConfig, configure_opencv_ffmpeg, read_credentials


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Previsualiza el substream RTSP de forma segura.")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--duration", type=float, default=0, help="Finalizar tras N segundos")
    parser.add_argument("--headless", action="store_true", help="No abrir una ventana")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.duration < 0:
        print("Error: duration no puede ser negativo.")
        return 2
    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        print(f"Error: {error}.")
        return 2

    configure_opencv_ffmpeg()
    config = CameraConfig()
    url = config.make_url(username, password)
    started = time.monotonic()
    samples: deque[float] = deque(maxlen=120)
    sequence = -1
    frames = 0
    width = height = 0
    print(f"Conectando con {config.name} ({config.host}, substream)...")

    try:
        with ReconnectingCamera(url) as camera:
            while True:
                now = time.monotonic()
                if args.duration and now - started >= args.duration:
                    break
                snapshot = camera.latest(sequence)
                if snapshot is None:
                    if not args.headless and cv2.waitKey(20) & 0xFF == ord("q"):
                        break
                    time.sleep(0.01)
                    continue
                sequence = snapshot.sequence
                frame = snapshot.frame
                frames += 1
                height, width = frame.shape[:2]
                samples.append(now)
                measured_fps = 0.0
                if len(samples) > 1:
                    measured_fps = (len(samples) - 1) / (samples[-1] - samples[0])
                if not args.headless:
                    cv2.putText(frame, f"FPS recibidos: {measured_fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.imshow("CameraCaptor - q para salir", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
    finally:
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    average = frames / elapsed if elapsed else 0.0
    print(f"Resolución recibida: {width}x{height}")
    print(f"Frames recibidos: {frames} en {elapsed:.1f} s")
    print(f"FPS medios observados: {average:.2f}")
    if frames == 0:
        print("Error: no se recibió ningún frame.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
