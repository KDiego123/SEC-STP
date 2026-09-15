"""Inferencia YOLO desacoplada para no bloquear la captura RTSP."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    coordinates: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    sequence: int
    detections: tuple[Detection, ...]
    inference_ms: float
    completed_at: float = 0.0


class YoloDetectorWorker:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence: float = 0.45,
        image_size: int = 640,
        allowed_classes: tuple[int, ...] = (0,),
        device: str = "auto",
    ) -> None:
        self.device = "0" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self._model_name = model_name
        self._model: YOLO | None = None
        self._confidence = confidence
        self._image_size = image_size
        self._allowed_classes = allowed_classes
        self._input: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=1)
        self._result: DetectionResult | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self.updates: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="yolo-inference", daemon=True)

    def start(self) -> "YoloDetectorWorker":
        self._thread.start()
        return self

    def submit(self, sequence: int, frame: np.ndarray) -> None:
        if not self._ready.is_set() or self._stop.is_set():
            return
        item = (sequence, frame.copy())
        try:
            self._input.put_nowait(item)
        except queue.Full:
            try:
                self._input.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input.put_nowait(item)
            except queue.Full:
                pass

    def latest(self) -> DetectionResult | None:
        with self._lock:
            return self._result

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    def _run(self) -> None:
        try:
            self._model = YOLO(self._model_name)
        except Exception as error:
            self.updates.put(f"YOLO: no se pudo cargar el modelo ({type(error).__name__}).")
            self._failed.set()
            return
        self._ready.set()
        self.updates.put(f"YOLO nano listo en {self.device}.")
        while not self._stop.is_set():
            try:
                sequence, frame = self._input.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                predictions = self._model.predict(
                    frame,
                    conf=self._confidence,
                    imgsz=self._image_size,
                    classes=list(self._allowed_classes),
                    device=self.device,
                    verbose=False,
                )
            except Exception as error:
                self.updates.put(f"YOLO: inferencia falló ({type(error).__name__}).")
                self._failed.set()
                return
            prediction = predictions[0]
            detections: list[Detection] = []
            for box in prediction.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=str(prediction.names[class_id]),
                        confidence=float(box.conf[0].item()),
                        coordinates=(x1, y1, x2, y2),
                    )
                )
            result = DetectionResult(
                sequence=sequence,
                detections=tuple(detections),
                inference_ms=float(prediction.speed.get("inference", 0.0)),
                completed_at=time.monotonic(),
            )
            with self._lock:
                self._result = result

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
