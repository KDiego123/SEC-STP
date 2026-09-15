"""Modelo y persistencia local de eventos de detección."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DetectionEvent:
    timestamp: str
    camera: str
    detected_class: str
    confidence: float
    coordinates: tuple[int, int, int, int]
    tracking_id: int | None = None

    @classmethod
    def create(
        cls,
        camera: str,
        detected_class: str,
        confidence: float,
        coordinates: tuple[int, int, int, int],
        tracking_id: int | None = None,
    ) -> "DetectionEvent":
        return cls(
            timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            camera=camera,
            detected_class=detected_class,
            confidence=confidence,
            coordinates=coordinates,
            tracking_id=tracking_id,
        )


class EventLogger:
    def __init__(self, path: Path = Path("events/events.jsonl")) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: DetectionEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(event), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def handle_detection(event: DetectionEvent) -> None:
    """Punto de extensión para futuras alertas e integraciones."""
    pass
