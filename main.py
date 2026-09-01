import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils.config_loader import load_config
from utils.sort_tracker import Sort
from video.stream import VideoStream


DETECTION_CLASSES = {
    0: ("Persona", (0, 255, 0)),
    2: ("Auto", (255, 170, 0)),
    3: ("Motocicleta", (0, 165, 255)),
    5: ("Bus", (255, 0, 255)),
    7: ("Camion", (0, 80, 255)),
}


def draw_interface(frame, counts, close_button):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    summary = "  |  ".join(
        f"{DETECTION_CLASSES[class_id][0]}: {counts[class_id]}"
        for class_id in DETECTION_CLASSES
    )
    cv2.putText(
        frame,
        summary,
        (14, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    left, top, right, bottom = close_button
    cv2.rectangle(frame, (left, top), (right, bottom), (30, 30, 210), -1)
    cv2.putText(
        frame,
        "CERRAR",
        (left + 13, top + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )


def main():
    config = load_config()
    camera_config = config["camera"]
    processing_config = config["processing"]
    display_config = config["display"]
    model_config = config["model"]

    model = YOLO(model_config["yolo_model"])
    trackers = {
        class_id: Sort(max_age=30, min_hits=1, iou_threshold=0.3)
        for class_id in DETECTION_CLASSES
    }
    stream = VideoStream(
        cam_type=camera_config["type"],
        source=camera_config["source"],
        reconnect_delay=camera_config["reconnect_delay"],
    )

    resize = processing_config.get("resize", 1.0)
    target_fps = processing_config.get("target_fps", 0)
    frame_delay = 1.0 / target_fps if target_fps > 0 else 0
    confidence = model_config.get("confidence_threshold", 0.4)
    show = display_config.get("show", True)
    window_name = display_config.get("window_name", "Deteccion de personas")

    close_requested = {"value": False}
    close_button = (0, 0, 0, 0)
    window_sized = False

    def handle_mouse(event, x, y, _flags, _params):
        if event != cv2.EVENT_LBUTTONUP:
            return
        left, top, right, bottom = close_button
        if left <= x <= right and top <= y <= bottom:
            close_requested["value"] = True

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, handle_mouse)

    print("[INFO] Vigilancia iniciada. Usa CERRAR, 'q' o Esc para salir.")

    try:
        while True:
            started_at = time.time()
            frame = stream.read()
            if frame is None:
                continue

            if resize != 1.0:
                frame = cv2.resize(frame, None, fx=resize, fy=resize)

            if show and not window_sized:
                cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])
                window_sized = True

            detections = {class_id: [] for class_id in DETECTION_CLASSES}
            results = model(
                frame,
                conf=confidence,
                classes=list(DETECTION_CLASSES),
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    detections[class_id].append([x1, y1, x2, y2, score])

            counts = {}
            for class_id, tracker in trackers.items():
                class_detections = detections[class_id]
                detection_array = (
                    np.asarray(class_detections, dtype=float)
                    if class_detections
                    else np.empty((0, 5), dtype=float)
                )
                tracks = tracker.update(detection_array)
                counts[class_id] = len(tracks)
                label, color = DETECTION_CLASSES[class_id]

                for track in tracks:
                    x1, y1, x2, y2, track_id = map(int, track)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} ID {track_id}",
                        (x1, max(72, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        color,
                        2,
                    )

            close_button = (frame.shape[1] - 105, 11, frame.shape[1] - 12, 47)
            draw_interface(frame, counts, close_button)

            if show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                if key in (ord("q"), 27) or close_requested["value"] or window_closed:
                    break

            elapsed = time.time() - started_at
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
