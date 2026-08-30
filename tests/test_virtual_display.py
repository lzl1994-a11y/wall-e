import unittest

from services.virtual_display import MjpegFrameSource, MjpegParser, VirtualDisplaySettings


class MjpegParserTests(unittest.TestCase):
    def test_handles_partial_and_multiple_frames(self):
        parser = MjpegParser()
        first = b"\xff\xd8one\xff\xd9"
        second = b"\xff\xd8two\xff\xd9"
        self.assertEqual(parser.feed(b"noise\xff\xd8on"), [])
        self.assertEqual(parser.feed(b"e\xff\xd9" + second), [first, second])

    def test_discards_noise_but_preserves_split_soi(self):
        parser = MjpegParser()
        self.assertEqual(parser.feed(b"noise\xff"), [])
        self.assertEqual(parser.feed(b"\xd8ok\xff\xd9"), [b"\xff\xd8ok\xff\xd9"])


class MjpegCommandTests(unittest.TestCase):
    def test_capture_command_uses_the_configured_display_geometry(self):
        settings = VirtualDisplaySettings(display_number=93, width=320, height=240, fps=20)
        command = MjpegFrameSource.command_for(settings)
        self.assertIn(":93.0", command)
        self.assertIn("320x240", command)
        self.assertIn("20", command)


if __name__ == "__main__":
    unittest.main()
