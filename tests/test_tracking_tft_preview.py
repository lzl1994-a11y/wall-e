import threading
import time
import unittest

from services.tracking_tft_preview import TrackingTftPreview


class _Result:
    busy = False
    last_frame = b"frame"


class _Server:
    def __init__(self):
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.calls = []

    def send_camera_preview(self, frame_provider, **kwargs):
        self.calls.append((frame_provider, kwargs))
        self.started.set()
        while not kwargs["should_stop"]():
            time.sleep(0.01)
        self.stopped.set()
        return _Result()


class TrackingTftPreviewTests(unittest.TestCase):
    def test_tracking_command_starts_and_stop_releases_stream(self):
        server = _Server()
        provider = object()
        preview = TrackingTftPreview(server, provider, fps=30)

        preview.set_command("start")
        self.assertTrue(server.started.wait(timeout=1.0))
        self.assertIs(server.calls[0][0], provider)
        self.assertEqual(server.calls[0][1]["duration_ms"], 60_000)
        self.assertEqual(server.calls[0][1]["hold_ms"], 0)
        self.assertEqual(server.calls[0][1]["fps"], 30)

        preview.set_command("stop")
        self.assertTrue(server.stopped.wait(timeout=1.0))
        self.assertFalse(preview.enabled)

    def test_pause_then_resume_restarts_when_tracking_remains_enabled(self):
        server = _Server()
        preview = TrackingTftPreview(server, object(), fps=10)
        preview.set_enabled(True)
        self.assertTrue(server.started.wait(timeout=1.0))

        self.assertTrue(preview.pause())
        self.assertTrue(server.stopped.wait(timeout=1.0))
        preview.resume()
        deadline = time.monotonic() + 1.0
        while len(server.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(server.calls), 2)
        preview.stop()


if __name__ == "__main__":
    unittest.main()
