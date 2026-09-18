"""Reconocimiento facial local con YuNet y SFace de OpenCV."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


MODEL_SPECS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceMatcherError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceMatch:
    name: str | None
    score: float
    coordinates: tuple[int, int, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(model_dir: Path) -> tuple[Path, Path]:
    """Descarga y valida los dos modelos oficiales si aún no existen."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, (url, expected_hash) in MODEL_SPECS.items():
        path = model_dir / filename
        if path.is_file() and _sha256(path) == expected_hash:
            paths.append(path)
            continue
        partial = path.with_suffix(path.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CameraCaptor/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as out:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 50_000_000:
                        raise FaceMatcherError("modelo facial demasiado grande")
                    out.write(chunk)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            raise FaceMatcherError(f"no se pudo descargar {filename}") from error
        if _sha256(partial) != expected_hash:
            partial.unlink(missing_ok=True)
            raise FaceMatcherError(f"la validación de {filename} falló")
        os.replace(partial, path)
        paths.append(path)
    return paths[0], paths[1]


class FaceMatcher:
    """Construye una plantilla por carpeta y compara rostros por coseno."""

    def __init__(self, gallery_dir: Path, model_dir: Path,
                 threshold: float = 0.50, min_face_size: int = 48) -> None:
        if not gallery_dir.is_dir():
            raise FaceMatcherError(f"galería inexistente: {gallery_dir}")
        if not 0.0 < threshold < 1.0 or min_face_size < 20:
            raise ValueError("parámetros faciales no válidos")
        detector_path, recognizer_path = ensure_models(model_dir)
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), 0.80, 0.30, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.gallery: dict[str, np.ndarray] = {}
        self.reference_count = 0
        self.skipped_count = 0
        self._load_gallery(gallery_dir)
        if not self.gallery:
            raise FaceMatcherError("ninguna fotografía produjo un rostro utilizable")

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray | None:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None

    def _detect(self, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        if width < 32 or height < 32:
            return None
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        return faces

    def _feature(self, image: np.ndarray, face: np.ndarray) -> np.ndarray | None:
        try:
            aligned = self.recognizer.alignCrop(image, face)
            return self._normalize(self.recognizer.feature(aligned))
        except cv2.error:
            return None

    @staticmethod
    def _largest(faces: np.ndarray | None) -> np.ndarray | None:
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda face: float(face[2] * face[3]))

    def _load_gallery(self, gallery_dir: Path) -> None:
        for person_dir in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
            features: list[np.ndarray] = []
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = cv2.imread(str(image_path))
                face = self._largest(self._detect(image)) if image is not None else None
                if face is None:
                    self.skipped_count += 1
                    continue
                feature = self._feature(image, face)
                if feature is None:
                    self.skipped_count += 1
                    continue
                features.append(feature)
                self.reference_count += 1
            if features:
                template = self._normalize(np.mean(features, axis=0))
                if template is not None:
                    self.gallery[person_dir.name] = template

    def match(self, person_crop: np.ndarray) -> FaceMatch | None:
        try:
            face = self._largest(self._detect(person_crop))
        except cv2.error:
            return None
        if face is None:
            return None
        x, y, width, height = (int(round(value)) for value in face[:4])
        image_height, image_width = person_crop.shape[:2]
        coordinates = (
            max(0, min(image_width - 1, x)),
            max(0, min(image_height - 1, y)),
            max(0, min(image_width - 1, x + width)),
            max(0, min(image_height - 1, y + height)),
        )
        if width < self.min_face_size or height < self.min_face_size:
            return FaceMatch(None, 0.0, coordinates)
        feature = self._feature(person_crop, face)
        if feature is None:
            return FaceMatch(None, 0.0, coordinates)
        scores = {name: float(np.dot(feature, template))
                  for name, template in self.gallery.items()}
        name, score = max(scores.items(), key=lambda item: item[1])
        return FaceMatch(name if score >= self.threshold else None, score, coordinates)
