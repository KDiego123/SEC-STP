import unittest

from src.camera_audio import command_from_result, normalize_speech


class CameraAudioTests(unittest.TestCase):
    def test_speech_is_normalized(self):
        self.assertEqual(normalize_speech("  ¡PRÉNDE las LUCES! "), "prende las luces")

    def test_exact_commands_are_mapped(self):
        self.assertEqual(
            command_from_result({
                "text": "prende las luces",
                "result": [{"word": "prende", "conf": 0.9},
                           {"word": "las", "conf": 0.9},
                           {"word": "luces", "conf": 0.9}],
            }),
            "voice_lights_on",
        )
        self.assertEqual(
            command_from_result({"text": "apaga las luces"}),
            "voice_lights_off",
        )

    def test_low_confidence_or_other_text_is_ignored(self):
        self.assertIsNone(command_from_result({
            "text": "prende las luces",
            "result": [{"word": "prende", "conf": 0.2}],
        }))
        self.assertIsNone(command_from_result({"text": "prende la luz"}))


if __name__ == "__main__":
    unittest.main()
