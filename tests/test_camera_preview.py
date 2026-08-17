import time
import unittest
from unittest.mock import patch

from services.camera_preview import CameraPreview


class _FakeImage:
    shape = (480, 640, 3)


class _FakeEncoded:
    def tobytes(self):
        return b"jpeg-frame"


class _FakeCapture:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def set(self, _prop, _value):
        return True

    def read(self):
        return True, _FakeImage()

    def release(self):
        self.released = True


class _FakeCv2:
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1

    def __init__(self, capture):
        self.capture = capture

    def VideoCapture(self, _device, _backend=None):
        return self.capture

    @staticmethod
    def imencode(_extension, _image, _options):
        return True, _FakeEncoded()


def _wait_for_state(preview, expected, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = preview.status()
        if status["state"] == expected:
            return status
        time.sleep(0.01)
    return preview.status()


class CameraPreviewTests(unittest.TestCase):
    def test_capture_runs_in_background_and_releases_device(self):
        capture = _FakeCapture()
        preview = CameraPreview("unused.yaml", frame_rate=20)
        with (
            patch("services.camera_preview.cv2", _FakeCv2(capture)),
            patch("services.camera_preview.resolve_camera_device", return_value="/dev/video2"),
        ):
            starting = preview.start()
            self.assertEqual(starting["state"], "starting")
            self.assertEqual(_wait_for_state(preview, "running")["device"], "/dev/video2")

            deadline = time.monotonic() + 1.0
            frame = None
            while time.monotonic() < deadline and frame is None:
                frame, _ = preview.get_frame()
                time.sleep(0.01)
            self.assertEqual(frame, b"jpeg-frame")

            stopped = preview.stop()
            self.assertEqual(stopped["state"], "stopped")
            self.assertTrue(capture.released)

    def test_missing_camera_is_reported_without_blocking_start(self):
        preview = CameraPreview("unused.yaml")
        with (
            patch("services.camera_preview.cv2", _FakeCv2(_FakeCapture())),
            patch("services.camera_preview.resolve_camera_device", return_value=None),
        ):
            started_at = time.monotonic()
            preview.start()
            self.assertLess(time.monotonic() - started_at, 0.2)
            status = _wait_for_state(preview, "error")
            self.assertEqual(status["state"], "error")
            self.assertIn("未找到", status["error"])


if __name__ == "__main__":
    unittest.main()
