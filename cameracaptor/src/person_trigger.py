"""Aviso de llegada tras un período observado sin personas."""

from __future__ import annotations

import time


class PersonArrivalTrigger:
    def __init__(self, idle_seconds: float = 12.0,
                 confirmations: int = 2, started_at: float | None = None) -> None:
        if idle_seconds <= 0 or confirmations < 1:
            raise ValueError("parámetros del trigger no válidos")
        self.idle_seconds = idle_seconds
        self.confirmations = confirmations
        self.reset(started_at)

    def reset(self, now: float | None = None) -> None:
        # Sin una inferencia negativa todavía, no existe inactividad observada.
        self._absence_since = now
        self._positive_count = 0
        self._eligible = False
        self._present = False

    def observe(self, present: bool, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        if not present:
            if self._present or self._absence_since is None:
                self._absence_since = moment
            self._present = False
            self._positive_count = 0
            self._eligible = False
            return False
        if not self._present:
            self._eligible = (self._absence_since is not None
                              and moment - self._absence_since >= self.idle_seconds)
            self._positive_count = 0
        self._present = True
        self._positive_count += 1
        if self._positive_count == self.confirmations and self._eligible:
            return True
        return False
