import unittest
import threading
import time
from unittest.mock import patch

from src.camera_speaker import (
    CameraSpeaker,
    SPEAKER_WAKE_AUDIO_SECONDS,
    SpeakerError,
)


class CameraSpeakerTests(unittest.TestCase):
    @patch("src.camera_speaker.send_pcmu")
    @patch("src.camera_speaker.synthesize_pcmu", return_value=b"\xaa" * 320)
    def test_idle_speaker_is_woken_before_phrase(self, synthesize, send):
        send.side_effect = lambda host, port, path, audio, stop: len(audio) // 160
        speaker = CameraSpeaker("192.0.2.1", 554, "/stream2")
        try:
            self.assertTrue(speaker.say("hola", tag="test"))
            result = speaker.results.get(timeout=3)
            self.assertTrue(result.succeeded)
            self.assertEqual(result.tag, "test")
            self.assertEqual(len(send.call_args_list), 2)
            self.assertEqual(send.call_args_list[0].args[3],
                             b"\xff" * int(8000 * SPEAKER_WAKE_AUDIO_SECONDS))
            self.assertEqual(send.call_args_list[1].args[3], b"\xaa" * 320)

            self.assertTrue(speaker.say("otra"))
            self.assertTrue(speaker.results.get(timeout=3).succeeded)
            self.assertEqual(len(send.call_args_list), 3)
            self.assertEqual(send.call_args_list[2].args[3], b"\xaa" * 320)
        finally:
            speaker.close()

    @patch("src.camera_speaker.synthesize_pcmu",
           side_effect=SpeakerError("fallo de prueba"))
    def test_failure_is_reported(self, synthesize):
        speaker = CameraSpeaker("192.0.2.1", 554, "/stream2")
        try:
            self.assertTrue(speaker.say("hola", tag="alerta"))
            result = speaker.results.get(timeout=2)
            self.assertFalse(result.succeeded)
            self.assertEqual(result.tag, "alerta")
        finally:
            speaker.close()

    @patch("src.camera_speaker.synthesize_pcmu", return_value=b"\xaa" * 1600)
    @patch("src.camera_speaker.send_pcmu")
    def test_tagged_audio_can_be_interrupted(self, send, synthesize):
        started = threading.Event()

        def cancellable_send(host, port, path, audio, stop):
            started.set()
            for _ in range(100):
                if stop.is_set():
                    return 1
                time.sleep(0.005)
            return len(audio) // 160

        send.side_effect = cancellable_send
        speaker = CameraSpeaker("192.0.2.1", 554, "/stream2")
        try:
            # Evita consumir tiempo en el precalentamiento durante esta prueba.
            speaker._last_playback_at = time.monotonic()
            self.assertTrue(speaker.say("aviso general", tag="persona"))
            self.assertTrue(started.wait(timeout=2))
            self.assertFalse(speaker.interrupt("otra-etiqueta"))
            self.assertTrue(speaker.interrupt("persona"))
            result = speaker.results.get(timeout=2)
            self.assertFalse(result.succeeded)
            self.assertTrue(result.interrupted)
            self.assertEqual(result.text, "aviso general")
        finally:
            speaker.close()

    @patch("src.camera_speaker.synthesize_pcmu")
    @patch("src.camera_speaker.send_pcmu")
    def test_replacement_runs_immediately_after_interruption(self, send, synthesize):
        first_started = threading.Event()
        calls = []
        synthesize.side_effect = lambda text: text.encode("utf-8") * 160

        def cancellable_send(host, port, path, audio, stop):
            calls.append(audio)
            if len(calls) == 1:
                first_started.set()
                while not stop.wait(0.005):
                    pass
                return 1
            return max(1, len(audio) // 160)

        send.side_effect = cancellable_send
        speaker = CameraSpeaker("192.0.2.1", 554, "/stream2")
        try:
            speaker._last_playback_at = time.monotonic()
            self.assertTrue(speaker.say("entrada", tag="persona"))
            self.assertTrue(first_started.wait(timeout=2))
            self.assertTrue(speaker.replace("Bienvenido Diego", tag="persona"))
            first = speaker.results.get(timeout=2)
            second = speaker.results.get(timeout=2)
            self.assertTrue(first.interrupted)
            self.assertEqual(second.text, "Bienvenido Diego")
            self.assertTrue(second.succeeded)
        finally:
            speaker.close()

    @patch("src.camera_speaker.send_pcmu")
    @patch("src.camera_speaker.synthesize_pcmu", return_value=b"\xaa" * 320)
    def test_repeated_phrase_uses_audio_cache(self, synthesize, send):
        send.side_effect = lambda host, port, path, audio, stop: len(audio) // 160
        speaker = CameraSpeaker("192.0.2.1", 554, "/stream2")
        try:
            speaker._last_playback_at = time.monotonic()
            self.assertTrue(speaker.say("Bienvenido Diego"))
            self.assertTrue(speaker.results.get(timeout=2).succeeded)
            self.assertTrue(speaker.say("Bienvenido Diego"))
            self.assertTrue(speaker.results.get(timeout=2).succeeded)
            self.assertEqual(synthesize.call_count, 1)
        finally:
            speaker.close()


if __name__ == "__main__":
    unittest.main()
