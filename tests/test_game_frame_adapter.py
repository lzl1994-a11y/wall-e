import threading
import time
import unittest

from services.game_frame_adapter import GameFrameAdapter


class _Stream:
    def __init__(self):
        self.frames = []
        self.started = threading.Event()
        self.release = threading.Event()

    def send_bgr(self, image):
        self.frames.append(int(image[0, 0, 0]))
        self.started.set()
        self.release.wait(timeout=1.0)
        return True


class GameFrameAdapterTests(unittest.TestCase):
    @staticmethod
    def _frame(value):
        return bytes([value, 0, 0, 0] * 4)

    def test_keeps_only_the_latest_frame_while_encoder_is_busy(self):
        stream = _Stream()
        adapter = GameFrameAdapter(stream, fps=60)
        adapter.submit_frame(self._frame(1), 2, 2, 8)
        self.assertTrue(stream.started.wait(timeout=1.0))
        adapter.submit_frame(self._frame(2), 2, 2, 8)
        adapter.submit_frame(self._frame(3), 2, 2, 8)
        stream.release.set()
        deadline = time.monotonic() + 1.0
        while len(stream.frames) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        adapter.close()
        self.assertEqual(stream.frames, [1, 3])
        self.assertEqual(adapter.callbacks, 3)
        self.assertEqual(adapter.overwritten, 1)

    def test_close_discards_a_pending_frame(self):
        stream = _Stream()
        adapter = GameFrameAdapter(stream, fps=1)
        adapter.submit_frame(self._frame(1), 2, 2, 8)
        self.assertTrue(stream.started.wait(timeout=1.0))
        adapter.submit_frame(self._frame(2), 2, 2, 8)
        adapter.close()
        stream.release.set()
        self.assertEqual(adapter.overwritten, 1)


if __name__ == "__main__":
    unittest.main()
