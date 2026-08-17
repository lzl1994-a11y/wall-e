"""Intent detection and one-shot frame capture through /camera_frame."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - ROS is unavailable in unit tests
    Image = None
    String = None
    qos_profile_sensor_data = 10

from services.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
    jpeg_from_ros_image,
)


def is_camera_inspection_request(user_prompt: str) -> bool:
    """快速识别常见的一次性视觉问题，持续注视/跟随不在此范围。"""
    text = (user_prompt or "").strip().lower()
    if not text or any(word in text for word in (
        "看着我", "看我", "盯着我", "跟着我", "跟随我", "look at me", "follow me"
    )):
        return False
    markers = (
        "看一下", "看一看", "帮我看看", "你看看", "请看看", "看下", "看一眼",
        "看什么", "是什么东西", "前面有什么", "面前有什么", "眼前有什么",
        "看到了什么", "看见了什么", "识别一下", "辨认一下", "认一下",
        "拍照", "拍张照", "拍一张", "照一张",
        "what is this", "what do you see", "take a photo",
    )
    return any(marker in text for marker in markers)


class CameraFrameProvider:
    """Request a leased camera session and wait for a fresh /camera_frame JPEG."""

    def __init__(self, node: Any, config_path: str | Path = "core/config.yaml"):
        del config_path  # Device selection belongs exclusively to camera_capture_node.
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._frame_time = 0.0
        self._status_state = ""
        self._status_time = 0.0
        self._status_error = ""
        self._command_pub = None
        self._subscriptions: list[Any] = []

        if Image is None or String is None:
            return
        self._command_pub = node.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        self._subscriptions = [
            node.create_subscription(
                Image,
                CAMERA_FRAME_TOPIC,
                self._on_image,
                qos_profile_sensor_data,
            ),
            node.create_subscription(String, CAMERA_STATUS_TOPIC, self._on_status, 10),
        ]

    def _on_image(self, message: Any) -> None:
        jpeg = jpeg_from_ros_image(message, quality=85)
        if not jpeg:
            return
        with self._condition:
            self._frame = jpeg
            self._frame_time = time.monotonic()
            self._condition.notify_all()

    def _on_status(self, message: Any) -> None:
        try:
            status = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        with self._condition:
            self._status_state = str(status.get("state", ""))
            self._status_error = str(status.get("error", ""))
            self._status_time = time.monotonic()
            self._condition.notify_all()

    def _publish_command(self, action: str, client_id: str, lease_sec: float = 0.0) -> None:
        if self._command_pub is None or String is None:
            return
        self._command_pub.publish(
            String(data=encode_camera_command(action, client_id, lease_sec))
        )

    def capture(
        self,
        timeout: float = 8.0,
        *,
        request_timeout: float | None = None,
    ) -> bytes | None:
        """Acquire the camera, return the first frame after this request, then release it."""
        if self._command_pub is None:
            return None
        frame_wait_seconds = max(0.2, float(timeout))
        request_wait_seconds = (
            frame_wait_seconds
            if request_timeout is None
            else max(0.2, float(request_timeout))
        )
        client_id = f"llm-{uuid.uuid4().hex}"
        requested_at = time.monotonic()
        deadline = requested_at + request_wait_seconds
        manager_acknowledged = False
        lease_seconds = request_wait_seconds + frame_wait_seconds + 2.0
        action = "acquire"
        try:
            with self._condition:
                while True:
                    if self._frame is not None and self._frame_time >= requested_at:
                        return self._frame
                    now = time.monotonic()
                    if (
                        not manager_acknowledged
                        and self._status_time >= requested_at
                        and self._status_state in {"starting", "streaming"}
                    ):
                        manager_acknowledged = True
                        deadline = now + frame_wait_seconds
                    remaining = deadline - now
                    if remaining <= 0:
                        return None
                    self._publish_command(action, client_id, lease_seconds)
                    action = "renew"
                    self._condition.wait(timeout=min(0.5, remaining))
        finally:
            self._publish_command("release", client_id)
