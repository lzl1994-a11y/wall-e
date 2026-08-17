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
    def __init__(self, messages, closed_event):
        self._messages = messages
        self._closed_event = closed_event

    def __iter__(self):
        for item in self._messages:
            delay, message = item if isinstance(item, tuple) else (0.0, item)
            if self._closed_event.wait(delay):
                return
            yield json.dumps(message, ensure_ascii=False) + "\n"
        self._closed_event.wait()


class _FakeProcess:
    def __init__(self, messages):
        self._closed_event = threading.Event()
        self.stdout = _FakeStdout(messages, self._closed_event)
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
        with patch.dict(os.environ, {"WALI_CAMERA_PREVIEW_PYTHON": sys.executable}):
            command = preview._worker_command()

        self.assertIn(sys.executable, command)
        self.assertIn(str(preview._frame_rate), command)
        self.assertNotIn("--device", command)

    def test_worker_defaults_to_the_config_web_python(self):
        preview = CameraPreview("unused.yaml")
        with patch.dict(os.environ, {}, clear=True):
            command = preview._worker_command()

        self.assertEqual(command[0], sys.executable)

    def test_preview_streams_only_from_camera_frame(self):
        process = _FakeProcess([
            {"type": "status", "phase": "requesting_camera", "source": "/camera_frame"},
            {"type": "status", "phase": "waiting_frame", "source": "/camera_frame"},
            {
                "type": "frame",
                "jpeg": base64.b64encode(b"jpeg-frame").decode("ascii"),
                "width": 640,
                "height": 480,
                "fps": 8.0,
                "source": "/camera_frame",
            },
        ])
        preview = CameraPreview("unused.yaml", frame_rate=20)
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            starting = preview.start()
            self.assertEqual(starting["state"], "starting")
            running = _wait_for_state(preview, "running")
            self.assertEqual(running["phase"], "streaming")
            self.assertEqual(running["source"], "/camera_frame")
            self.assertEqual(running["device"], "")

            frame, status = preview.get_frame()
            self.assertEqual(frame, b"jpeg-frame")
            self.assertEqual((status["width"], status["height"]), (640, 480))

            stopped = preview.stop()
            self.assertEqual(stopped["state"], "stopped")
            self.assertTrue(process.terminated)

    def test_worker_error_is_reported(self):
        process = _FakeProcess([{
            "type": "error",
            "error": "camera_capture_node 不可用",
        }])
        preview = CameraPreview("unused.yaml")
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            preview.start()
            status = _wait_for_state(preview, "error")

        self.assertIn("camera_capture_node", status["error"])
        self.assertTrue(process.terminated)

    def test_missing_camera_frame_times_out_with_manager_diagnostic(self):
        process = _FakeProcess([{
            "type": "status",
            "phase": "waiting_frame",
            "source": "/camera_frame",
            "diagnostic": "hobot_usb_cam 已退出，退出码 1",
        }])
        preview = CameraPreview("unused.yaml", startup_timeout=0.2)
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            preview.start()
            status = _wait_for_state(preview, "error", timeout=1.0)

        self.assertIn("等待 /camera_frame 超时", status["error"])
        self.assertIn("hobot_usb_cam 已退出", status["error"])
        self.assertTrue(process.terminated)

    def test_manager_ack_starts_a_fresh_first_frame_deadline(self):
        process = _FakeProcess([
            {"type": "status", "phase": "requesting_camera"},
            (0.22, {"type": "status", "phase": "waiting_frame"}),
            (0.16, {
                "type": "frame",
                "jpeg": base64.b64encode(b"delayed-frame").decode("ascii"),
                "width": 640,
                "height": 480,
                "source": "/camera_frame",
            }),
        ])
        preview = CameraPreview(
            "unused.yaml",
            request_timeout=0.35,
            startup_timeout=0.25,
        )
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            preview.start()
            status = _wait_for_state(preview, "running", timeout=1.0)

        self.assertEqual(status["state"], "running")
        frame, _status = preview.get_frame()
        self.assertEqual(frame, b"delayed-frame")
        preview.stop()

    def test_repeated_waiting_status_does_not_extend_first_frame_timeout(self):
        process = _FakeProcess([
            {"type": "status", "phase": "waiting_frame"},
            (0.07, {"type": "status", "phase": "waiting_frame"}),
            (0.07, {"type": "status", "phase": "waiting_frame"}),
            (0.07, {"type": "status", "phase": "waiting_frame"}),
        ])
        preview = CameraPreview("unused.yaml", startup_timeout=0.12)
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            preview.start()
            status = _wait_for_state(preview, "error", timeout=0.3)

        self.assertEqual(status["state"], "error")
        self.assertIn("等待 /camera_frame 超时", status["error"])

    def test_camera_manager_request_has_its_own_timeout(self):
        process = _FakeProcess([])
        preview = CameraPreview(
            "unused.yaml",
            request_timeout=0.2,
            startup_timeout=0.8,
        )
        with patch("services.camera_preview.subprocess.Popen", return_value=process):
            preview.start()
            status = _wait_for_state(preview, "error", timeout=0.6)

        self.assertIn("等待 camera_capture_node 响应超时", status["error"])


if __name__ == "__main__":
    unittest.main()
