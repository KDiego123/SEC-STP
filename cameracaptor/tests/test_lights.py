import unittest

from src.lights import LightsWorker


class FakeCameraWeb:
    def __init__(self):
        self.lamp = 0
        self.night = 0
        self.calls = []

    def get(self, module, channel=0):
        if module == "lamp_panel":
            return {"lamp_mode": self.lamp}
        if module == "image":
            return {"day_night_mode": self.night}
        raise AssertionError(module)

    def set_day_night_mode(self, mode, channel=0):
        self.calls.append(("night", mode))
        self.night = mode

    def set_lamp_mode(self, mode, channel=0):
        self.calls.append(("lamp", mode))
        self.lamp = mode


class LightsWorkerTests(unittest.TestCase):
    @staticmethod
    def _next_update(worker):
        return worker.updates.get(timeout=2)

    def test_voice_on_forces_night_before_white_lamp(self):
        camera = FakeCameraWeb()
        worker = LightsWorker(camera)
        try:
            self._next_update(worker)  # Lectura inicial.
            self.assertTrue(worker.operate("voice_lights_on"))
            message, lamp, night = self._next_update(worker)
            self.assertEqual(camera.calls, [("night", 2), ("lamp", 1)])
            self.assertEqual((lamp, night), (1, 2))
            self.assertIn("activados", message)
        finally:
            worker.close()

    def test_voice_off_keeps_night_and_selects_ir(self):
        camera = FakeCameraWeb()
        camera.lamp = 1
        camera.night = 2
        worker = LightsWorker(camera)
        try:
            self._next_update(worker)
            self.assertTrue(worker.operate("voice_lights_off"))
            message, lamp, night = self._next_update(worker)
            self.assertEqual(camera.calls, [("lamp", 0)])
            self.assertEqual((lamp, night), (0, 2))
            self.assertIn("apagados", message)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
