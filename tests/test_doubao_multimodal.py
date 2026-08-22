import unittest

from services.multimodal.doubao_multimodal import DoubaoMultimodal


class DoubaoMultimodalTests(unittest.TestCase):
    def test_builds_ark_chat_audio_content(self):
        message = DoubaoMultimodal().build_audio_message("aGVsbG8=")

        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][1], {
            "type": "input_audio",
            "input_audio": {"data": "aGVsbG8=", "format": "wav"},
        })
