"""Aviso de llegada tras un período observado sin personas."""

from __future__ import annotations

import time
from dataclasses import dataclass


ENTRY_ALERT = "Se ha identificado a alguien en la entrada"


def welcome_alert(name: str) -> str:
    identity = name.strip()
    if not identity:
        raise ValueError("la identidad no puede estar vacía")
    return f"Bienvenido {identity}"


@dataclass(frozen=True)
class PersonAnnouncement:
    kind: str
    text: str
    identity: str | None = None


class PersonAlertSequence:
    """Genera un aviso de entrada y luego una bienvenida por identidad."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._entry_scheduled = False
        self._welcomes_scheduled: set[str] = set()

    def arm(self) -> None:
        self.reset()
        self._active = True

    def announcements(
        self, names: list[str] | tuple[str, ...]
    ) -> tuple[PersonAnnouncement, ...]:
        if not self._active:
            return ()
        pending: list[PersonAnnouncement] = []
        if not self._entry_scheduled:
            pending.append(PersonAnnouncement("entry", ENTRY_ALERT))
            self._entry_scheduled = True
        unique = sorted({name.strip() for name in names if name.strip()})
        for name in unique:
            if name not in self._welcomes_scheduled:
                pending.append(PersonAnnouncement("welcome", welcome_alert(name), name))
                self._welcomes_scheduled.add(name)
        return tuple(pending)


class PersonArrivalTrigger:
    def __init__(self, idle_seconds: float = 12.0,
                 confirmation_seconds: float = 1.5,
                 started_at: float | None = None) -> None:
        if idle_seconds <= 0 or confirmation_seconds <= 0:
            raise ValueError("parámetros del trigger no válidos")
        self.idle_seconds = idle_seconds
        self.confirmation_seconds = confirmation_seconds
        self.reset(started_at)

    def reset(self, now: float | None = None) -> None:
        # Sin una inferencia negativa todavía, no existe inactividad observada.
        self._absence_since = now
        self._presence_since: float | None = None
        self._eligible = False
        self._present = False
        self._alerted = False

    def idle_elapsed(self, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        return (not self._present and self._absence_since is not None
                and moment - self._absence_since >= self.idle_seconds)

    def observe(self, present: bool, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        if not present:
            if self._present or self._absence_since is None:
                self._absence_since = moment
            self._present = False
            self._presence_since = None
            self._eligible = False
            self._alerted = False
            return False
        if not self._present:
            self._eligible = (self._absence_since is not None
                              and moment - self._absence_since >= self.idle_seconds)
            self._presence_since = moment
            self._alerted = False
        self._present = True
        if (self._eligible and not self._alerted and self._presence_since is not None
                and moment - self._presence_since >= self.confirmation_seconds):
            self._alerted = True
            return True
        return False
