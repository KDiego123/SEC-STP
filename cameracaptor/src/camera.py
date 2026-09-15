"""Captura RTSP con lectura desacoplada y reconexión automática."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class FrameSnapshot:
    frame: object
    sequence: int
    captured_at: float


class ReconnectingCamera:
    def __init__(self, url: str, reconnect_delay: float = 2.0) -> None:
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: FrameSnapshot | None = None
        self._sequence = 0
        self._connected = False

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def start(self) -> "ReconnectingCamera":
        self._thread = threading.Thread(target=self._run, name="rtsp-capture", daemon=True)
        self._thread.start()
        return self

    def latest(self, after_sequence: int = -1) -> FrameSnapshot | None:
        with self._lock:
            item = self._latest
            if item is None or item.sequence <= after_sequence:
                return None
            return FrameSnapshot(item.frame.copy(), item.sequence, item.captured_at)

    def _open(self) -> bool:
        self._capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._capture.isOpened()

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._lock:
            self._connected = False

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._open():
                self._release()
                self._stop.wait(self._reconnect_delay)
                continue
            with self._lock:
                self._connected = True
            while not self._stop.is_set() and self._capture is not None:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    break
                with self._lock:
                    self._sequence += 1
                    self._latest = FrameSnapshot(frame, self._sequence, time.monotonic())
            self._release()
            self._stop.wait(self._reconnect_delay)
        self._release()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                # Solo fuerza el cierre si el timeout de FFmpeg no desbloqueó read().
                self._release()
                self._thread.join(timeout=1)

    def __enter__(self) -> "ReconnectingCamera":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
