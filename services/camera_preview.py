"""On-demand camera preview used by the local configuration page."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - optional on development hosts
    cv2 = None

try:
    from services.usb_devices import resolve_camera_device
except ImportError:  # Supports: python services/web_server.py
    from usb_devices import resolve_camera_device


class CameraPreview:
    """Capture and cache JPEG frames without blocking HTTP request threads."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        idle_timeout: float = 15.0,
        frame_rate: float = 8.0,
    ) -> None:
        self._config_path = Path(config_path)
        self._idle_timeout = max(2.0, float(idle_timeout))
        self._frame_interval = 1.0 / max(1.0, float(frame_rate))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._state = "stopped"
        self._error = ""
        self._device = ""
        self._frame: bytes | None = None
        self._frame_time = 0.0
        self._last_client_time = 0.0
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._last_client_time = time.monotonic()
            if self._thread is not None and self._thread.is_alive():
                return self._status_locked()

            self._generation += 1
            generation = self._generation
            self._stop_event = threading.Event()
            self._state = "starting"
            self._error = ""
            self._device = ""
            self._frame = None
            self._frame_time = 0.0
            self._width = 0
            self._height = 0
            self._fps = 0.0
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(generation, self._stop_event),
                name="camera-preview",
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def stop(self, *, wait: bool = True) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            if thread is not None and thread.is_alive():
                self._state = "stopping"
            else:
                self._state = "stopped"
                self._thread = None

        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        with self._lock:
            if thread is None or not thread.is_alive():
                self._state = "stopped"
                self._thread = None
            return self._status_locked()

    def close(self) -> None:
        self.stop(wait=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def get_frame(self) -> tuple[bytes | None, dict[str, Any]]:
        with self._lock:
            self._last_client_time = time.monotonic()
            frame = self._frame if self._state == "running" else None
            return frame, self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        age_ms = None
        if self._frame_time:
            age_ms = max(0, int((time.monotonic() - self._frame_time) * 1000))
        return {
            "state": self._state,
            "device": self._device,
            "width": self._width,
            "height": self._height,
            "fps": round(self._fps, 1),
            "frame_age_ms": age_ms,
            "error": self._error,
        }

    def _set_error(self, generation: int, message: str) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._state = "error"
            self._error = message

    def _capture_loop(self, generation: int, stop_event: threading.Event) -> None:
        capture = None
        ended_with_error = False
        try:
            if cv2 is None:
                raise RuntimeError("当前环境未安装 OpenCV，无法启动摄像头预览")

            device = resolve_camera_device(self._config_path)
            if not device:
                raise RuntimeError("未找到已配置的摄像头设备")

            backend = getattr(cv2, "CAP_V4L2", None) if str(device).startswith("/dev/video") else None
            capture = cv2.VideoCapture(device, backend) if backend is not None else cv2.VideoCapture(device)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开摄像头 {device}，设备可能正被视觉跟踪占用")

            for prop_name, value in (
                ("CAP_PROP_FRAME_WIDTH", 640),
                ("CAP_PROP_FRAME_HEIGHT", 480),
                ("CAP_PROP_BUFFERSIZE", 1),
            ):
                prop = getattr(cv2, prop_name, None)
                if prop is not None:
                    capture.set(prop, value)

            with self._lock:
                if generation != self._generation:
                    return
                self._state = "running"
                self._device = str(device)

            sample_started = time.monotonic()
            sample_frames = 0
            measured_fps = 0.0
            failed_reads = 0
            while not stop_event.is_set():
                now = time.monotonic()
                with self._lock:
                    if now - self._last_client_time > self._idle_timeout:
                        stop_event.set()
                        break

                frame_started = time.monotonic()
                ok, image = capture.read()
                if not ok or image is None:
                    failed_reads += 1
                    if failed_reads >= 20:
                        raise RuntimeError("摄像头连续读帧失败")
                    stop_event.wait(0.05)
                    continue

                failed_reads = 0
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                if not encoded_ok:
                    continue

                sample_frames += 1
                elapsed = time.monotonic() - sample_started
                if elapsed >= 1.0:
                    measured_fps = sample_frames / elapsed
                    sample_started = time.monotonic()
                    sample_frames = 0

                height, width = image.shape[:2]
                with self._lock:
                    if generation != self._generation:
                        return
                    self._frame = encoded.tobytes()
                    self._frame_time = time.monotonic()
                    self._width = int(width)
                    self._height = int(height)
                    self._fps = measured_fps

                remaining = self._frame_interval - (time.monotonic() - frame_started)
                if remaining > 0:
                    stop_event.wait(remaining)
        except Exception as exc:
            ended_with_error = True
            self._set_error(generation, str(exc))
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                if generation == self._generation:
                    if not ended_with_error:
                        self._state = "stopped"
                    self._thread = None
