import unittest
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


if __name__ == "__main__":
    unittest.main()
