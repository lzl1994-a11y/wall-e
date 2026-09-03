import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.tracking_tft_preview import TrackingTftPreview


class _Stream:
    def __init__(self):
        self.closed = False
        self.frames = []
        self.stopped = threading.Event()

    def send_encoded_jpeg(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        if not self.closed:
            self.closed = True
            self.stopped.set()


class _Server:
    def __init__(self):
        self.settings = SimpleNamespace(jpeg_quality=70)
        self.started = threading.Event()
        self.streams = []
        self.fps_values = []

    def open_persistent_stream(self, *, fps):
        stream = _Stream()
        self.streams.append(stream)
        self.fps_values.append(fps)
        self.started.set()
        return stream


class _Provider:
    def __init__(self):
        self.calls = []

    def capture_stream(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["on_source_frame"](b"camera-jpeg", 1)
        while not kwargs["should_stop"]():
            time.sleep(0.01)
        return b"camera-jpeg"


class TrackingTftPreviewTests(unittest.TestCase):
    @patch("services.tracking_tft_preview.prepare_tft_jpeg", side_effect=lambda frame, **_: frame)
    def test_tracking_uses_one_persistent_stream_until_stop(self, _prepare):
        server = _Server()
        provider = _Provider()
        preview = TrackingTftPreview(server, provider, fps=30)

        preview.set_command("start")
        self.assertTrue(server.started.wait(timeout=1.0))
        deadline = time.monotonic() + 1.0
        while not server.streams[0].frames and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(server.fps_values, [30])
        self.assertEqual(server.streams[0].frames, [b"camera-jpeg"])
        self.assertEqual(provider.calls[0]["duration_ms"], 60_000)
        self.assertEqual(provider.calls[0]["fps"], 30)

        preview.set_command("stop")
        self.assertTrue(server.streams[0].stopped.wait(timeout=1.0))
        self.assertFalse(preview.enabled)

    @patch("services.tracking_tft_preview.prepare_tft_jpeg", side_effect=lambda frame, **_: frame)
    def test_pause_then_resume_opens_a_new_persistent_session(self, _prepare):
        server = _Server()
        preview = TrackingTftPreview(server, _Provider(), fps=10)
        preview.set_enabled(True)
        self.assertTrue(server.started.wait(timeout=1.0))

        self.assertTrue(preview.pause())
        self.assertTrue(server.streams[0].stopped.wait(timeout=1.0))
        preview.resume()
        deadline = time.monotonic() + 1.0
        while len(server.streams) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(server.streams), 2)
        preview.stop()


if __name__ == "__main__":
    unittest.main()
