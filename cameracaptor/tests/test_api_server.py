import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api_server import (
    PanelApiBridge,
    create_api_app,
    load_or_create_api_token,
)


class ApiServerTests(unittest.TestCase):
    def setUp(self):
        self.bridge = PanelApiBridge()
        self.token = "token-de-prueba-suficientemente-largo"
        self.client = TestClient(create_api_app(self.bridge, self.token))
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_protected_endpoints_require_token(self):
        self.assertEqual(self.client.get("/api/v1/status").status_code, 401)
        response = self.client.get("/api/v1/status", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("camera_connected", response.json())

    def test_tts_is_queued_and_returns_accepted(self):
        response = self.client.post(
            "/api/v1/tts", headers=self.headers, json={"text": "Hola"}
        )
        self.assertEqual(response.status_code, 202)
        command = self.bridge.commands.get_nowait()
        self.assertEqual(command.action, "tts")
        self.assertEqual(command.payload, {"text": "Hola"})

    def test_snapshot_and_events_are_exposed(self):
        self.assertEqual(
            self.client.get("/api/v1/snapshot.jpg", headers=self.headers).status_code,
            503,
        )
        self.bridge.set_snapshot(b"jpeg-test")
        snapshot = self.client.get("/api/v1/snapshot.jpg", headers=self.headers)
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.content, b"jpeg-test")
        event = self.bridge.add_event("person_detected", {"people": 1})
        events = self.client.get(
            f"/api/v1/events?after={event['id'] - 1}", headers=self.headers
        ).json()["events"]
        self.assertEqual(events, [event])

    def test_token_is_created_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_token.txt"
            first = load_or_create_api_token(path)
            second = load_or_create_api_token(path)
            self.assertEqual(first, second)
            self.assertGreaterEqual(len(first), 24)


if __name__ == "__main__":
    unittest.main()
