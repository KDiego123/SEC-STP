"""API HTTP local que comparte el estado y los controles del panel."""

from __future__ import annotations

import asyncio
import copy
import hmac
import queue
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


API_VERSION = "1.0"
MAX_EVENTS = 250


class ApiServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiCommand:
    command_id: str
    action: str
    payload: dict[str, Any]
    created_at: str


class PanelApiBridge:
    """Intercambio seguro entre FastAPI y el hilo principal de Tkinter."""

    def __init__(self) -> None:
        self.commands: queue.Queue[ApiCommand] = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._status: dict[str, Any] = {
            "camera_connected": False,
            "detection_available": False,
            "detection_enabled": False,
            "people": 0,
            "identities": [],
            "speaker_busy": False,
            "lights_busy": False,
            "lamp_mode": None,
            "night_mode": None,
            "ptz_available": False,
        }
        self._snapshot: bytes | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._next_event_id = 1

    def submit(self, action: str, payload: dict[str, Any] | None = None) -> ApiCommand:
        command = ApiCommand(
            command_id=uuid.uuid4().hex,
            action=action,
            payload=payload or {},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self.commands.put_nowait(command)
        except queue.Full as error:
            raise ApiServerError("la cola de control está ocupada") from error
        self.add_event("command_accepted", {
            "command_id": command.command_id,
            "action": action,
        })
        return command

    def update_status(self, **values: Any) -> None:
        with self._lock:
            self._status.update(copy.deepcopy(values))

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._status)
        result["uptime_seconds"] = round(time.monotonic() - self._started_at, 1)
        result["api_version"] = API_VERSION
        return result

    def set_snapshot(self, content: bytes) -> None:
        with self._lock:
            self._snapshot = content

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._snapshot

    def add_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            event = {
                "id": self._next_event_id,
                "type": event_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data": copy.deepcopy(data),
            }
            self._next_event_id += 1
            self._events.append(event)
            return copy.deepcopy(event)

    def events(self, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            selected = [item for item in self._events if item["id"] > after]
            return copy.deepcopy(selected[-limit:])


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class NightModeRequest(BaseModel):
    mode: Literal["auto", "day", "night"]


class PtzMoveRequest(BaseModel):
    direction: Literal["up", "down", "left", "right"]
    speed: float = Field(default=0.3, ge=0.1, le=0.6)
    duration_ms: int = Field(default=400, ge=100, le=3000)


class DetectionRequest(BaseModel):
    enabled: bool


def load_or_create_api_token(path: Path) -> str:
    try:
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if len(token) < 24:
                raise ApiServerError("el token existente es demasiado corto")
            return token
        token = secrets.token_urlsafe(32)
        path.write_text(token + "\n", encoding="utf-8")
        return token
    except OSError as error:
        raise ApiServerError("no se pudo leer o crear el token de la API") from error


def create_api_app(bridge: PanelApiBridge, token: str) -> FastAPI:
    app = FastAPI(
        title="CameraCaptor API",
        version=API_VERSION,
        description="Control de la cámara administrada por el panel CameraCaptor.",
    )
    bearer = HTTPBearer(auto_error=False)

    def authorize(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        supplied = credentials.credentials if credentials is not None else ""
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(
                status_code=401,
                detail="token inválido o ausente",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def accepted(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            command = bridge.submit(action, payload)
        except ApiServerError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "accepted": True,
            "command_id": command.command_id,
            "action": command.action,
        }

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "CameraCaptor API", "docs": "/docs"}

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        status = bridge.status()
        return {
            "service": "ok",
            "camera_connected": status["camera_connected"],
            "uptime_seconds": status["uptime_seconds"],
        }

    @app.get("/api/v1/status", dependencies=[Depends(authorize)])
    def status() -> dict[str, Any]:
        return bridge.status()

    @app.get("/api/v1/snapshot.jpg", dependencies=[Depends(authorize)])
    def snapshot() -> Response:
        content = bridge.snapshot()
        if content is None:
            raise HTTPException(status_code=503, detail="todavía no hay un fotograma")
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/events", dependencies=[Depends(authorize)])
    def events(after: int = 0, limit: int = 100) -> dict[str, Any]:
        safe_limit = min(max(limit, 1), MAX_EVENTS)
        return {"events": bridge.events(max(after, 0), safe_limit)}

    @app.post(
        "/api/v1/tts", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def tts(request: TtsRequest) -> dict[str, Any]:
        return accepted("tts", {"text": request.text.strip()})

    @app.post(
        "/api/v1/lights/on", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def lights_on() -> dict[str, Any]:
        return accepted("lights_on")

    @app.post(
        "/api/v1/lights/off", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def lights_off() -> dict[str, Any]:
        return accepted("lights_off")

    @app.post(
        "/api/v1/night-mode", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def night_mode(request: NightModeRequest) -> dict[str, Any]:
        return accepted("night_mode", {"mode": request.mode})

    @app.post(
        "/api/v1/ptz/move", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def ptz_move(request: PtzMoveRequest) -> dict[str, Any]:
        return accepted("ptz_move", request.model_dump())

    @app.post(
        "/api/v1/ptz/stop", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def ptz_stop() -> dict[str, Any]:
        return accepted("ptz_stop")

    @app.post(
        "/api/v1/detection", status_code=202,
        dependencies=[Depends(authorize)],
    )
    def detection(request: DetectionRequest) -> dict[str, Any]:
        return accepted("detection", {"enabled": request.enabled})

    @app.websocket("/api/v1/events/live")
    async def live_events(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            authentication = await asyncio.wait_for(
                websocket.receive_json(), timeout=5.0
            )
            supplied = str(authentication.get("token", ""))
            if not hmac.compare_digest(supplied, token):
                await websocket.close(code=1008, reason="token inválido")
                return
            last_id = max(int(authentication.get("after", 0)), 0)
            await websocket.send_json({"type": "ready", "api_version": API_VERSION})
            next_heartbeat = time.monotonic() + 15.0
            while True:
                pending = bridge.events(last_id, MAX_EVENTS)
                for event in pending:
                    await websocket.send_json(event)
                    last_id = event["id"]
                if time.monotonic() >= next_heartbeat:
                    await websocket.send_json({"type": "heartbeat"})
                    next_heartbeat = time.monotonic() + 15.0
                await asyncio.sleep(0.25)
        except Exception:
            # Una desconexión del cliente no debe afectar el panel.
            return

    return app


class PanelApiServer:
    def __init__(self, bridge: PanelApiBridge, host: str, port: int,
                 token_file: Path) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("puerto de API fuera del intervalo válido")
        self.bridge = bridge
        self.host = host
        self.port = port
        self.token_file = token_file
        self.token = load_or_create_api_token(token_file)
        self.app = create_api_app(bridge, self.token)
        self.updates: queue.Queue[str] = queue.Queue()
        self._server = None
        self._thread = threading.Thread(
            target=self._run, name="cameracaptor-api", daemon=True
        )

    @property
    def display_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> "PanelApiServer":
        self._thread.start()
        self.updates.put(f"API iniciando en {self.display_url}")
        return self

    def _run(self) -> None:
        try:
            import uvicorn

            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self._server.run()
            if not self._server.started:
                self.updates.put("API: no se pudo abrir el puerto configurado.")
        except Exception as error:
            self.updates.put(f"API no disponible: {type(error).__name__}.")

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=5)
