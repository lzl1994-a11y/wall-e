"""Shared protocol and helpers for the hot-standby ROS camera."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


CAMERA_FRAME_TOPIC = "/camera_frame"
CAMERA_COMMAND_TOPIC = "/camera_capture_cmd"
CAMERA_STATUS_TOPIC = "/camera_capture_status"
# The one canonical stream produced by the only ``hobot_usb_cam`` process.
# Tracking consumes it directly; the capture manager adapts it to the public
# CompressedImage preview topic for legacy preview/photo consumers.
CAMERA_SOURCE_TOPIC = "/image"
TRACKING_IMAGE_TOPIC = CAMERA_SOURCE_TOPIC  # Backwards-compatible import name.
DEFAULT_ROS_SETUP = Path("/opt/tros/humble/setup.bash")


def encode_camera_command(action: str, client_id: str, lease_sec: float = 0.0) -> str:
    payload: dict[str, Any] = {
        "action": str(action),
        "client_id": str(client_id),
    }
    if lease_sec > 0:
        payload["lease_sec"] = float(lease_sec)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_camera_command(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action", "")).strip().lower()
    client_id = str(payload.get("client_id", "")).strip()
    if action not in {"acquire", "renew", "release"} or not client_id:
        return None
    try:
        lease_sec = float(payload.get("lease_sec", 5.0))
    except (TypeError, ValueError):
        lease_sec = 5.0
    return {
        "action": action,
        "client_id": client_id[:96],
        "lease_sec": min(30.0, max(1.0, lease_sec)),
    }


class CameraLeaseBook:
    """Track expiring camera clients without coupling to ROS or wall-clock time."""

    def __init__(self) -> None:
        self._leases: dict[str, float] = {}

    def acquire(self, client_id: str, lease_sec: float, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else float(now)
        self._leases[client_id] = current + min(30.0, max(1.0, float(lease_sec)))

    def release(self, client_id: str) -> None:
        self._leases.pop(client_id, None)

    def purge(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else float(now)
        expired = [client_id for client_id, deadline in self._leases.items() if deadline <= current]
        for client_id in expired:
            self._leases.pop(client_id, None)

    @property
    def active(self) -> bool:
        return bool(self._leases)

    @property
    def count(self) -> int:
        return len(self._leases)


class CameraWatchdogAction(str, Enum):
    """Actions produced by the camera capture watchdog decision."""

    FIRST_FRAME_TIMEOUT = "first_frame_timeout"
    FRAME_TIMEOUT = "frame_timeout"
    START_PROCESS = "start_process"
    PUBLISH_STATUS = "publish_status"
    NONE = "none"


@dataclass(frozen=True)
class CameraWatchdogDecision:
    """Pure decision outcome produced by the camera watchdog check."""

    action: CameraWatchdogAction
    state: str | None = None
    elapsed_sec: float = 0.0


DEFAULT_FIRST_FRAME_TIMEOUT_SEC = 45.0
DEFAULT_FRAME_TIMEOUT_SEC = 3.0
DEFAULT_OUTPUT_ACTIVE_WINDOW_SEC = 1.0


def evaluate_camera_watchdog(
    *,
    now: float,
    process_alive: bool = False,
    process_started_at: float = 0.0,
    last_source_frame: float = 0.0,
    last_output_frame: float = 0.0,
    has_active_leases: bool = False,
    retry_after: float = 0.0,
    first_frame_timeout_sec: float = DEFAULT_FIRST_FRAME_TIMEOUT_SEC,
    frame_timeout_sec: float = DEFAULT_FRAME_TIMEOUT_SEC,
    output_active_window_sec: float = DEFAULT_OUTPUT_ACTIVE_WINDOW_SEC,
) -> CameraWatchdogDecision:
    """Evaluate camera process watchdog and status transitions.

    Pure decision logic:
    1. If process is alive:
       - If last_source_frame < process_started_at:
         first frame timeout when now - process_started_at > first_frame_timeout_sec.
       - If last_source_frame >= process_started_at:
         stream stall when now - last_source_frame > frame_timeout_sec.
       - Otherwise, normal status selection:
         - has_active_leases: 'streaming' if now - last_output_frame <= output_active_window_sec else 'starting'
         - no active leases: 'standby' if last_source_frame >= process_started_at else 'starting'
    2. If process is not alive:
       - start process when now >= retry_after
       - otherwise no action (waiting for retry delay)
    """
    if process_alive:
        if last_source_frame < process_started_at:
            waited = now - process_started_at
            if waited > first_frame_timeout_sec:
                return CameraWatchdogDecision(
                    action=CameraWatchdogAction.FIRST_FRAME_TIMEOUT,
                    elapsed_sec=waited,
                )
        else:
            stalled = now - last_source_frame
            if stalled > frame_timeout_sec:
                return CameraWatchdogDecision(
                    action=CameraWatchdogAction.FRAME_TIMEOUT,
                    elapsed_sec=stalled,
                )

        if has_active_leases:
            state = (
                "streaming"
                if now - last_output_frame <= output_active_window_sec
                else "starting"
            )
        else:
            state = (
                "standby"
                if last_source_frame >= process_started_at
                else "starting"
            )
        return CameraWatchdogDecision(
            action=CameraWatchdogAction.PUBLISH_STATUS,
            state=state,
        )

    if now >= retry_after:
        return CameraWatchdogDecision(action=CameraWatchdogAction.START_PROCESS)

    return CameraWatchdogDecision(action=CameraWatchdogAction.NONE)


def build_hobot_camera_command(
    device: str,
    *,
    ros_setup: str | Path | None = DEFAULT_ROS_SETUP,
) -> list[str]:
    command = [
        "ros2",
        "run",
        "hobot_usb_cam",
        "hobot_usb_cam",
        "--ros-args",
        "--log-level",
        "WARN",
        "-p",
        f"video_device:={device}",
        "-p",
        "image_width:=640",
        "-p",
        "image_height:=480",
        "-p",
        # The ESP UVC camera exposes 640x480 MJPEG at 15 fps.  The driver's
        # 30 fps default is not a supported mode and never produces /image.
        "framerate:=15",
    ]
    setup_path = str(ros_setup or "").strip()
    if os.name != "nt" and setup_path and Path(setup_path).is_file():
        return [
            "bash",
            "-c",
            'source "$1" && exec "${@:2}"',
            "camera-capture",
            setup_path,
            *command,
        ]
    return command


def jpeg_from_ros_image(
    message: Any,
    *,
    quality: int = 85,
    validate_decode: bool = True,
) -> bytes | None:
    """Convert an Image or CompressedImage payload to JPEG bytes."""
    encoding = str(getattr(message, "encoding", "")).lower()
    image_format = str(getattr(message, "format", "")).lower()
    raw = bytes(getattr(message, "data", b""))
    if (
        encoding in {"jpeg", "jpg", "mjpeg"}
        or "jpeg" in image_format
        or "jpg" in image_format
        or raw.startswith(b"\xff\xd8")
    ):
        # A camera that has just been opened can briefly publish an incomplete
        # MJPEG buffer.  The ROS message still labels it as JPEG, but cloud
        # vision APIs reject it with an image parsing error.  Require a complete
        # JPEG envelope and, when OpenCV is available, make sure it decodes.
        if not raw.startswith(b"\xff\xd8"):
            return None
        eoi = raw.rfind(b"\xff\xd9")
        if eoi < 2:
            return None
        jpeg = raw[:eoi + 2]
        if not validate_decode:
            return jpeg
        try:
            import cv2
            import numpy as np
        except ImportError:
            return jpeg
        try:
            decoded = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        except Exception:
            return None
        if decoded is None or decoded.size == 0:
            return None
        return jpeg

    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    channels = 1 if encoding in {"mono8", "8uc1"} else 3
    expected = width * height * channels
    if width <= 0 or height <= 0 or len(raw) < expected:
        return None
    image = np.frombuffer(raw, dtype=np.uint8)[:expected].reshape((height, width, channels))
    if channels == 1:
        image = image[:, :, 0]
    elif encoding.startswith("rgb"):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return encoded.tobytes() if ok else None
