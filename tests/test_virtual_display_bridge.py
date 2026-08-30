import threading
import time
import unittest
from unittest.mock import Mock

from services.virtual_display import VirtualDisplaySettings
from services.virtual_display_bridge import VirtualDisplayTftBridge


class _FakeDisplay:
    def __init__(self):
        self.start = Mock()
        self.launch = Mock()
        self.stop = Mock()


class _FakeFrames:
    def __init__(self):
        self.start = Mock()
        self.stop = Mock()
        self._latest = (0, None)

    @property
    def latest(self):
        return self._latest


class _FakeStream:
    def __init__(self, result=True):
        self.result = result
        self.frames = []
        self.closed = False

    def send_jpeg(self, jpeg):
        self.frames.append(jpeg)
        return self.result

    def close(self):
        self.closed = True


class VirtualDisplayTftBridgeTests(unittest.TestCase):
    def setUp(self):
        self.display = _FakeDisplay()
        self.frames = _FakeFrames()
        self.server = Mock()
        self.settings = VirtualDisplaySettings(fps=100)

    def test_bridge_forwards_only_newest_sequences_and_cleans_up(self):
        stream = _FakeStream()
        self.server.open_jpeg_stream.return_value = stream
        bridge = VirtualDisplayTftBridge(
            self.server, settings=self.settings, display=self.display, frames=self.frames
        )

        bridge.start(["demo-app"])
        self.frames._latest = (1, b"first")
        deadline = time.monotonic() + 0.5
        while not stream.frames and time.monotonic() < deadline:
            time.sleep(0.01)
        self.frames._latest = (1, b"duplicate")
        time.sleep(0.03)
        self.frames._latest = (2, b"latest")
        deadline = time.monotonic() + 0.5
        while len(stream.frames) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        bridge.stop()

        self.assertEqual(stream.frames, [b"first", b"latest"])
        self.assertTrue(stream.closed)
        self.display.stop.assert_called_once()
        self.frames.stop.assert_called_once()

    def test_unavailable_tft_prevents_application_start(self):
        self.server.open_jpeg_stream.return_value = None
        bridge = VirtualDisplayTftBridge(
            self.server, settings=self.settings, display=self.display, frames=self.frames
        )

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            bridge.start(["demo-app"])

        self.display.launch.assert_not_called()
        self.display.stop.assert_called_once()
        self.frames.stop.assert_called_once()

    def test_stream_loss_notifies_owner(self):
        stream = _FakeStream(result=False)
        self.server.open_jpeg_stream.return_value = stream
        lost = threading.Event()
        bridge = VirtualDisplayTftBridge(
            self.server,
            settings=self.settings,
            display=self.display,
            frames=self.frames,
            on_stream_lost=lost.set,
        )

        bridge.start(["demo-app"])
        self.frames._latest = (1, b"frame")
        self.assertTrue(lost.wait(timeout=0.5))
        bridge.stop()


if __name__ == "__main__":
    unittest.main()
