"""Síntesis de voz local y no bloqueante mediante Windows SAPI."""

from __future__ import annotations

import queue
import subprocess
import threading


class WindowsSpeaker:
    def __init__(self) -> None:
        self._messages: queue.Queue[str | None] = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._run, name="local-tts", daemon=True)
        self._thread.start()

    def say(self, message: str) -> None:
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            pass

    def _run(self) -> None:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice.Speak($args[0])"
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            message = self._messages.get()
            if message is None:
                return
            try:
                subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", script, message],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                    creationflags=creation_flags,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue

    def close(self) -> None:
        try:
            self._messages.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)
