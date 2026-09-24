"""Acceso al panel de lámparas de la interfaz web de la cámara."""

from __future__ import annotations

import hashlib
import json
import secrets
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request


class LightError(RuntimeError):
    pass


class CameraWeb:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.session_id: str | None = None

    def _request(self, query: dict[str, str] | None = None,
                 payload: dict[str, object] | None = None,
                 headers: dict[str, str] | None = None) -> tuple[dict[str, object], object]:
        url = f"http://{self.host}/cgi-bin/web.cgi"
        if query is not None:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers or {})
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                result = json.loads(response.read(100_000))
                response_headers = response.headers
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise LightError("la interfaz web de la cámara no respondió") from error
        if not isinstance(result, dict):
            raise LightError("respuesta web inválida")
        return result, response_headers

    def login(self) -> None:
        info, _ = self._request({"mod": "session", "cmd": "get_auth_info"})
        if info.get("status") == "error":
            raise LightError("la cámara no proporcionó parámetros de autenticación")
        realm, nonce, qop = (info.get(key) for key in ("realm", "nonce", "qop"))
        if not all(isinstance(item, str) and item for item in (realm, nonce, qop)):
            raise LightError("autenticación web no compatible")
        cnonce = secrets.token_hex(8)
        md5 = lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()
        uri = "/cgi-bin/web.cgi?mod=account&cmd=check"
        ha1 = md5(f"{self.username}:{realm}:{self.password}")
        ha2 = md5(f"GET:{uri}")
        digest = md5(f"{ha1}:{nonce}:00000001:{cnonce}:{qop}:{ha2}")
        auth = (f'Digest username="{self.username}",realm="{realm}",nonce="{nonce}",'
                f'uri="{uri}",cnonce="{cnonce}",nc=00000001,qop="{qop}",'
                f'response="{digest}"')
        result, headers = self._request(
            {"mod": "session", "cmd": "login1"}, headers={"Authorization": auth})
        session_id = headers.get("Session-Id")
        if result.get("status") != "ok" or not session_id:
            raise LightError("credenciales web rechazadas")
        self.session_id = session_id

    def get(self, mod: str, channel: int = 0) -> dict[str, object]:
        query = {"mod": mod, "cmd": "get"}
        if mod != "device":
            query["param2"] = json.dumps({"channel": channel}, separators=(",", ":"))
        for attempt in range(2):
            if self.session_id is None:
                self.login()
            result, _ = self._request(query, headers={"Session-Id": self.session_id or ""})
            if result.get("status") == "expired" and attempt == 0:
                self.session_id = None
                continue
            break
        if result.get("status") in ("error", "expired"):
            raise LightError(f"la cámara rechazó la consulta {mod}")
        return result

    def set_brightness(self, value: int, channel: int = 0) -> None:
        if not 0 <= value <= 100:
            raise ValueError("brillo fuera de 0-100")
        payload = {"mod": "image", "cmd": "set_single",
                   "param": {"channel": channel},
                   "param2": {"cmd": 5, "value": value}}
        for attempt in range(2):
            if self.session_id is None:
                self.login()
            result, _ = self._request(payload=payload,
                                      headers={"Session-Id": self.session_id or ""})
            if result.get("status") == "expired" and attempt == 0:
                self.session_id = None
                continue
            break
        if result.get("status") in ("error", "expired"):
            raise LightError("la cámara rechazó el cambio de brillo")
        current = self.get("image", channel).get("max_led_brightness")
        if current != value:
            raise LightError("la cámara no confirmó el nuevo brillo")

    def set_day_night_mode(self, mode: int, channel: int = 0) -> None:
        if mode not in (0, 1, 2, 3):
            raise ValueError("modo día/noche no válido")
        self.set_image_options({"day_night_mode": mode}, channel)

    def set_image_options(self, updates: dict[str, int], channel: int = 0) -> None:
        allowed = {"day_night_mode", "max_led_brightness", "led_brightness_mode"}
        if not updates or set(updates) - allowed:
            raise ValueError("opciones de imagen no válidas")
        if "day_night_mode" in updates and updates["day_night_mode"] not in (0, 1, 2, 3):
            raise ValueError("modo día/noche no válido")
        if "max_led_brightness" in updates and not 0 <= updates["max_led_brightness"] <= 100:
            raise ValueError("brillo fuera de 0-100")
        if "led_brightness_mode" in updates and updates["led_brightness_mode"] not in (0, 1):
            raise ValueError("modo de brillo LED no válido")
        current = self.get("image", channel)
        fields = (
            "mirror", "flip", "nr_enable", "wdr_enable", "backlight_mode",
            "anti_flicker_enable", "anti_flicker_mode", "ldc_enable", "face_mode",
            "smart_face_mode", "brightness", "saturation", "sharpness", "contrast",
            "drc_strenght", "max_led_brightness", "led_brightness_mode",
            "day_begin", "day_end", "day_night_mode", "ircut_delay", "tv_standard",
            "exposure", "day_night_lux", "night_fps_select", "rotation",
            "day_to_night_brightness", "night_to_day_brightness",
        )
        if any(field not in current for field in fields):
            raise LightError("faltan ajustes de imagen para cambiar el modo")
        image = {field: current[field] for field in fields}
        image.update(updates)
        payload = {"mod": "image", "cmd": "set",
                   "param": {"channel": channel}, "param2": image}
        for attempt in range(2):
            if self.session_id is None:
                self.login()
            result, _ = self._request(payload=payload,
                                      headers={"Session-Id": self.session_id or ""})
            if result.get("status") == "expired" and attempt == 0:
                self.session_id = None
                continue
            break
        if result.get("status") != "ok":
            raise LightError("la cámara rechazó los ajustes de imagen")
        confirmed = self.get("image", channel)
        if any(confirmed.get(key) != value for key, value in updates.items()):
            raise LightError("la cámara no confirmó los ajustes de imagen")

    def set_lamp_mode(self, mode: int, channel: int = 0) -> None:
        if mode not in (0, 1, 2):
            raise ValueError("modo de lámparas no válido")
        current = self.get("lamp_panel", channel)
        switch = current.get("day_night_switch")
        if switch not in (0, 1):
            raise LightError("conmutador día/noche no disponible")
        payload = {"mod": "lamp_panel", "cmd": "set",
                   "param": {"channel": channel},
                   "param2": {"lamp_mode": mode, "day_night_switch": switch}}
        for attempt in range(2):
            if self.session_id is None:
                self.login()
            result, _ = self._request(payload=payload,
                                      headers={"Session-Id": self.session_id or ""})
            if result.get("status") == "expired" and attempt == 0:
                self.session_id = None
                continue
            break
        if result.get("status") in ("error", "expired"):
            raise LightError("la cámara rechazó el modo de lámparas")
        if self.get("lamp_panel", channel).get("lamp_mode") != mode:
            raise LightError("la cámara no confirmó el modo de lámparas")


class LightsWorker:
    """Cambia lámpara y día/noche por separado sin bloquear Tkinter."""

    def __init__(self, controller: CameraWeb) -> None:
        self.controller = controller
        self.updates: queue.Queue[tuple[str, int | None, int | None]] = queue.Queue()
        self._commands: queue.Queue[str] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._busy = threading.Event()
        self._lamp: int | None = None
        self._night: int | None = None
        self._baseline: tuple[int, int] | None = None
        self._thread = threading.Thread(target=self._run, name="lights-control", daemon=True)
        self._thread.start()
        self._commands.put_nowait("read")

    @property
    def lamp_mode(self) -> int | None:
        return self._lamp

    @property
    def night_mode(self) -> int | None:
        return self._night

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def operate(self, command: str) -> bool:
        if command not in ("lamp_on", "lamp_off", "lamp_auto",
                           "night_on", "night_off", "night_auto", "read",
                           "voice_lights_on", "voice_lights_off"):
            raise ValueError("orden de modos no válida")
        if self._closed.is_set() or self._busy.is_set():
            return False
        self._busy.set()
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def refresh(self) -> bool:
        return self.operate("read")

    def _read_state(self) -> str:
        lamp = self.controller.get("lamp_panel").get("lamp_mode")
        day = self.controller.get("image").get("day_night_mode")
        if not isinstance(lamp, int) or lamp not in (0, 1, 2) or day not in (0, 1, 2, 3):
            raise LightError("modos de lámparas e imagen no disponibles")
        if self._baseline is None:
            self._baseline = (lamp, day) if day not in (1, 2) else (2, 0)
        self._lamp, self._night = lamp, day
        lamp_label = {0: "IR", 1: "blanca", 2: "mixta"}[lamp]
        day_label = {0: "automática", 1: "diurna", 2: "nocturna", 3: "programada"}[day]
        return f"Lámpara {lamp_label}; visión {day_label}."

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                command = self._commands.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if command == "read":
                    message = self._read_state()
                elif command == "lamp_on":
                    self.controller.set_lamp_mode(1)
                    self._read_state()
                    message = ("Focos blancos encendidos en modo nocturno."
                               if self._night == 2 else
                               "Lámpara blanca seleccionada; requiere modo nocturno para iluminar.")
                elif command == "lamp_off":
                    self.controller.set_lamp_mode(0)
                    self._read_state()
                    message = "Focos blancos apagados; IR disponible para visión nocturna."
                elif command == "lamp_auto":
                    lamp = (self._baseline or (2, 0))[0]
                    self.controller.set_lamp_mode(lamp)
                    self._read_state()
                    message = "Modo de lámpara original restaurado."
                elif command == "night_on":
                    self.controller.set_day_night_mode(2)
                    self._read_state()
                    message = "Visión nocturna activada."
                elif command == "night_off":
                    self.controller.set_day_night_mode(1)
                    self._read_state()
                    message = "Visión nocturna desactivada; el firmware apaga los focos en día."
                elif command == "voice_lights_on":
                    # El firmware solo ilumina con la lámpara blanca si la imagen
                    # está forzada a noche; la secuencia debe ser atómica.
                    self.controller.set_day_night_mode(2)
                    self.controller.set_lamp_mode(1)
                    self._read_state()
                    message = "Comando de voz: visión nocturna y focos blancos activados."
                elif command == "voice_lights_off":
                    # Apaga la iluminación y fuerza el modo diurno para salir
                    # completamente de la visión nocturna.
                    self.controller.set_lamp_mode(0)
                    self.controller.set_day_night_mode(1)
                    self._read_state()
                    message = "Comando de voz: focos apagados y visión diurna activada."
                else:
                    day = (self._baseline or (2, 0))[1]
                    self.controller.set_day_night_mode(day)
                    self._read_state()
                    message = "Modo día/noche original restaurado."
                self.updates.put((message, self._lamp, self._night))
            except (LightError, ValueError) as error:
                try:
                    self._read_state()
                except LightError:
                    pass
                self.updates.put((f"Modos: {error}", self._lamp, self._night))
            finally:
                self._busy.clear()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=5)
