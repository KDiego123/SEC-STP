"""Inferencia YOLO desacoplada para no bloquear la captura RTSP."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.face_matcher import FaceMatcher, FaceMatcherError


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    coordinates: tuple[int, int, int, int]
    identity: str | None = None
    identity_score: float = 0.0
    face_coordinates: tuple[int, int, int, int] | None = None


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
        face_gallery: str | None = None,
        face_model_dir: str | None = None,
        face_threshold: float = 0.50,
    ) -> None:
        self.device = "0" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self._model_name = model_name
        self._model: YOLO | None = None
        self._confidence = confidence
        self._image_size = image_size
        self._allowed_classes = allowed_classes
        self._face_gallery = face_gallery
        self._face_model_dir = face_model_dir
        self._face_threshold = face_threshold
        self._face_matcher: FaceMatcher | None = None
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
            self.updates.put("YOLO nano cargado; preparando detección rápida...")
            # La primera inferencia inicializa internamente el modelo y puede tardar
            # varios segundos. Hacerla aquí evita cargar ese coste a la primera
            # persona que aparezca frente a la cámara.
            self._model.predict(
                np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8),
                conf=self._confidence,
                imgsz=self._image_size,
                classes=list(self._allowed_classes),
                device=self.device,
                verbose=False,
            )
            if self._face_gallery and self._face_model_dir:
                self.updates.put("Cargando galería de reconocimiento facial...")
                try:
                    self._face_matcher = FaceMatcher(
                        Path(self._face_gallery), Path(self._face_model_dir),
                        self._face_threshold)
                    self.updates.put(
                        f"Rostros: {len(self._face_matcher.gallery)} identidad(es), "
                        f"{self._face_matcher.reference_count} foto(s) útil(es), "
                        f"{self._face_matcher.skipped_count} omitida(s)."
                    )
                except (FaceMatcherError, cv2.error, OSError, ValueError) as error:
                    self.updates.put(
                        f"Reconocimiento facial desactivado: {type(error).__name__}.")
                    self._face_matcher = None
        except Exception as error:
            self.updates.put(
                f"YOLO: no se pudo cargar o preparar el modelo ({type(error).__name__})."
            )
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
                identity = None
                identity_score = 0.0
                face_coordinates = None
                if self._face_matcher is not None:
                    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size:
                        match = self._face_matcher.match(crop)
                        if match is not None:
                            identity = match.name
                            identity_score = match.score
                            fx1, fy1, fx2, fy2 = match.coordinates
                            face_coordinates = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=str(prediction.names[class_id]),
                        confidence=float(box.conf[0].item()),
                        coordinates=(x1, y1, x2, y2),
                        identity=identity,
                        identity_score=identity_score,
                        face_coordinates=face_coordinates,
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
