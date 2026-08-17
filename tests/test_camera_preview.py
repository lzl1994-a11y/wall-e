import base64
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from services.camera_preview import CameraPreview


class _FakeStdout:
    def __init__(self, lines, closed_event):
        self._lines = lines
        self._closed_event = closed_event

    def __iter__(self):
        yield from self._lines
        self._closed_event.wait()


class _FakeProcess:
    def __init__(self, messages):
        self._closed_event = threading.Event()
        lines = [json.dumps(message, ensure_ascii=False) + "\n" for message in messages]
        self.stdout = _FakeStdout(lines, self._closed_event)
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0
        self._closed_event.set()

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        if self.returncode is None and not self._closed_event.wait(timeout):
            raise subprocess.TimeoutExpired("fake-camera", timeout)
        return self.returncode


def _wait_for_state(preview, expected, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = preview.status()
        if status["state"] == expected:
            return status
        time.sleep(0.01)
    return preview.status()


class CameraPreviewTests(unittest.TestCase):
    def test_worker_uses_configured_ros_python_when_available(self):
        preview = CameraPreview("unused.yaml")
        with patch.dict(os.environ, {"WALI_ROS_PYTHON": sys.executable}):
            command = preview._worker_command("/dev/video0")

        self.assertIn(sys.executable, command)
        self.assertIn(str(preview._frame_rate), command)

    def test_capture_runs_in_subprocess_and_releases_device(self):
        process = _FakeProcess([
            {"type": "status", "phase": "opening"},
            {"type": "status", "phase": "waiting_frame"},
            {
                "type": "frame",
                "jpeg": base64.b64encode(b"jpeg-frame").decode("ascii"),
                "width": 640,
                "height": 480,
                "fps": 8.0,
                "source": "/image_padded_jpeg",
            },
        ])
        preview = CameraPreview("unused.yaml", frame_rate=20)
        with (
            patch("services.camera_preview.resolve_camera_device", return_value="/dev/video2"),
            patch("services.camera_preview.subprocess.Popen", return_value=process),
        ):
            starting = preview.start()
            self.assertEqual(starting["state"], "starting")
            running = _wait_for_state(preview, "running")
            self.assertEqual(running["device"], "/dev/video2")
            self.assertEqual(running["phase"], "streaming")
            self.assertEqual(running["source"], "/image_padded_jpeg")

            frame, status = preview.get_frame()
            self.assertEqual(frame, b"jpeg-frame")
            self.assertEqual((status["width"], status["height"]), (640, 480))

            stopped = preview.stop()
            self.assertEqual(stopped["state"], "stopped")
            self.assertTrue(process.terminated)

    def test_missing_camera_is_reported_without_blocking_start(self):
        preview = CameraPreview("unused.yaml")
        with patch("services.camera_preview.resolve_camera_device", return_value=None):
            started_at = time.monotonic()
            preview.start()
            self.assertLess(time.monotonic() - started_at, 0.2)
            status = _wait_for_state(preview, "error")
            self.assertEqual(status["state"], "error")
            self.assertIn("未找到", status["error"])

    def test_blocked_camera_process_is_terminated_after_startup_timeout(self):
        process = _FakeProcess([{
            "type": "status",
            "phase": "opening",
            "diagnostic": "ROS Python 环境不可用: rclpy not installed",
        }])
        preview = CameraPreview("unused.yaml", startup_timeout=0.2)
        with (
            patch("services.camera_preview.resolve_camera_device", return_value="/dev/video0"),
            patch("services.camera_preview.subprocess.Popen", return_value=process),
        ):
            preview.start()
            status = _wait_for_state(preview, "error", timeout=1.0)
            self.assertEqual(status["state"], "error")
            self.assertEqual(status["phase"], "error")
            self.assertIn("打开摄像头 /dev/video0 超时", status["error"])
            self.assertIn("ROS Python 环境不可用", status["error"])
            self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
