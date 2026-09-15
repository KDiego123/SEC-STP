"""Control PTZ ONVIF con parada explícita y timeout de movimiento."""

from __future__ import annotations

import queue
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
PTZ_NS = "http://www.onvif.org/ver20/ptz/wsdl"
MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA_NS = "http://www.onvif.org/ver10/schema"


class PtzError(RuntimeError):
    pass


def _soap(host: str, endpoint: str, body: str) -> ET.Element:
    envelope = (
        f'<s:Envelope xmlns:s="{SOAP_NS}" xmlns:tptz="{PTZ_NS}" '
        f'xmlns:trt="{MEDIA_NS}" xmlns:tt="{SCHEMA_NS}">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    )
    request = urllib.request.Request(
        f"http://{host}/onvif/{endpoint}",
        data=envelope.encode("utf-8"),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = response.read(100_000)
    except urllib.error.HTTPError as error:
        raise PtzError(f"ONVIF respondió HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError) as error:
        raise PtzError("ONVIF no respondió") from error
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise PtzError("respuesta ONVIF inválida") from error
    fault = root.find(f".//{{{SOAP_NS}}}Fault")
    if fault is not None:
        raise PtzError("la cámara rechazó el comando PTZ")
    return root


def _substream_token(host: str) -> str:
    root = _soap(host, "Media", "<trt:GetProfiles/>")
    profiles = root.findall(f".//{{{MEDIA_NS}}}Profiles")
    for profile in profiles:
        name = profile.find(f"{{{SCHEMA_NS}}}Name")
        if name is not None and name.text == "SubStream":
            return profile.attrib["token"]
    if profiles:
        return profiles[0].attrib["token"]
    raise PtzError("ONVIF no devolvió perfiles de video")


class OnvifPtz:
    def __init__(self, host: str) -> None:
        self.host = host
        self.profile_token = _substream_token(host)

    def move(self, direction: str, speed: float = 0.3) -> None:
        if not 0.1 <= speed <= 0.6:
            raise ValueError("velocidad PTZ fuera del intervalo seguro")
        x, y = {"left": (-speed, 0.0), "right": (speed, 0.0),
                "up": (0.0, speed), "down": (0.0, -speed)}[direction]
        body = (
            "<tptz:ContinuousMove>"
            f"<tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>"
            f'<tptz:Velocity><tt:PanTilt x="{x:.2f}" y="{y:.2f}"/></tptz:Velocity>'
            "<tptz:Timeout>PT0.4S</tptz:Timeout>"
            "</tptz:ContinuousMove>"
        )
        _soap(self.host, "Ptz", body)

    def stop(self) -> None:
        body = (
            "<tptz:Stop>"
            f"<tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>"
            "<tptz:PanTilt>true</tptz:PanTilt>"
            "</tptz:Stop>"
        )
        _soap(self.host, "Ptz", body)


class PtzWorker:
    """Serializa movimientos; una parada elimina movimientos pendientes."""

    def __init__(self, controller: OnvifPtz) -> None:
        self.controller = controller
        self.updates: queue.Queue[str] = queue.Queue()
        self._commands: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ptz-control", daemon=True)
        self._thread.start()

    def move(self, direction: str, speed: float) -> None:
        if self._closed.is_set():
            return
        try:
            self._commands.put_nowait((direction, speed))
        except queue.Full:
            pass

    def stop(self) -> None:
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        try:
            self._commands.put_nowait(("stop", 0.0))
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._closed.is_set() or not self._commands.empty():
            try:
                direction, speed = self._commands.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if direction == "stop":
                    self.controller.stop()
                else:
                    self.controller.move(direction, speed)
            except (PtzError, ValueError) as error:
                self.updates.put(f"PTZ: {error}")

    def close(self) -> None:
        self.stop()
        self._closed.set()
        self._thread.join(timeout=5)
