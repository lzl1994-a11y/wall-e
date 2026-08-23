"""Intent detection and one-shot frame capture through /camera_frame."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

try:
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - ROS is unavailable in unit tests
    CompressedImage = None
    String = None
    qos_profile_sensor_data = 10

from services.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
    jpeg_from_ros_image,
)


def is_camera_photo_request(user_prompt: str) -> bool:
    """识别只拍照保存、不调用视觉模型的请求。"""
    text = (user_prompt or "").strip().lower()
    if not text:
        return False
    markers = (
        "拍照", "拍张照", "拍一张", "照张相", "照一张", "给我拍", "帮我拍",
        "摄影", "take a photo", "take a picture",
    )
    return any(marker in text for marker in markers)


def is_camera_inspection_request(user_prompt: str) -> bool:
    """快速识别常见的一次性视觉问题，持续注视/跟随不在此范围。"""
    text = (user_prompt or "").strip().lower()
    if not text or is_camera_photo_request(text) or any(word in text for word in (
        "看着我", "看我", "盯着我", "跟着我", "跟随我", "look at me", "follow me"
    )):
        return False
    markers = (
        "看一下", "看一看", "帮我看看", "你看看", "请看看", "看下", "看一眼",
        "看什么", "是什么东西", "前面有什么", "面前有什么", "眼前有什么",
        "看到了什么", "看见了什么", "识别一下", "辨认一下", "认一下",
        "what is this", "what do you see",
    )
    return any(marker in text for marker in markers)


class CameraFrameProvider:
    """Request a leased camera session and wait for a fresh /camera_frame JPEG."""

    DEFAULT_WARMUP_SECONDS = 0.5
    DEFAULT_WARMUP_FRAMES = 3

    def __init__(
        self,
        node: Any,
        config_path: str | Path = "core/config.yaml",
        *,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    ):
        del config_path  # Device selection belongs exclusively to camera_capture_node.
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._frame_time = 0.0
        self._frame_sequence = 0
        self._warmup_seconds = max(0.0, float(warmup_seconds))
        self._warmup_frames = max(1, int(warmup_frames))
        self._status_state = ""
        self._status_time = 0.0
        self._status_error = ""
        self._command_pub = None
        self._subscriptions: list[Any] = []

        if CompressedImage is None or String is None:
            return
        self._command_pub = node.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        self._subscriptions = [
            node.create_subscription(
                CompressedImage,
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
            self._frame_sequence += 1
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
        first_fresh_frame_time = 0.0
        first_fresh_frame_sequence = 0
        lease_seconds = request_wait_seconds + frame_wait_seconds + 2.0
        action = "acquire"
        try:
            with self._condition:
                while True:
                    if self._frame is not None and self._frame_time >= requested_at:
                        if not first_fresh_frame_time:
                            first_fresh_frame_time = self._frame_time
                            first_fresh_frame_sequence = self._frame_sequence
                        warmup_elapsed = self._frame_time - first_fresh_frame_time
                        warmup_frame_count = (
                            self._frame_sequence - first_fresh_frame_sequence + 1
                        )
                        if (
                            warmup_elapsed >= self._warmup_seconds
                            and warmup_frame_count >= self._warmup_frames
                        ):
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

    def capture_stream(
        self,
        *,
        duration_ms: int,
        fps: int,
        on_frame: Callable[[bytes], None],
        on_source_frame: Callable[[bytes, int], None] | None = None,
        on_no_new_frame: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        timeout: float = 8.0,
        request_timeout: float | None = None,
    ) -> bytes | None:
        """Lease the shared camera and deliver paced, complete JPEG frames.

        The preview clock starts only after a fresh, warmed-up frame arrives, so
        camera startup time does not shorten the requested TFT preview.
        """
        if self._command_pub is None:
            return None
        frame_wait_seconds = max(0.2, float(timeout))
        request_wait_seconds = (
            frame_wait_seconds
            if request_timeout is None
            else max(0.2, float(request_timeout))
        )
        duration_seconds = max(0.1, float(duration_ms) / 1000.0)
        target_fps = min(30, max(1, int(fps)))
        frame_interval = 1.0 / target_fps
        client_id = f"tft-{uuid.uuid4().hex}"
        requested_at = time.monotonic()
        deadline = requested_at + request_wait_seconds
        manager_acknowledged = False
        first_fresh_frame_time = 0.0
        first_fresh_frame_sequence = 0
        lease_seconds = request_wait_seconds + frame_wait_seconds + duration_seconds + 2.0
        action = "acquire"
        first_frame: bytes | None = None

        try:
            with self._condition:
                while first_frame is None:
                    if should_stop is not None and should_stop():
                        return None
                    if self._frame is not None and self._frame_time >= requested_at:
                        if not first_fresh_frame_time:
                            first_fresh_frame_time = self._frame_time
                            first_fresh_frame_sequence = self._frame_sequence
                        warmup_elapsed = self._frame_time - first_fresh_frame_time
                        warmup_frame_count = (
                            self._frame_sequence - first_fresh_frame_sequence + 1
                        )
                        if (
                            warmup_elapsed >= self._warmup_seconds
                            and warmup_frame_count >= self._warmup_frames
                        ):
                            first_frame = self._frame
                            break
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

            stream_started = time.monotonic()
            stream_deadline = stream_started + duration_seconds
            next_frame_at = stream_started
            last_renewed_at = stream_started
            last_frame = first_frame
            last_delivered_sequence = -1

            while True:
                if should_stop is not None and should_stop():
                    break
                now = time.monotonic()
                if now >= stream_deadline:
                    break
                wait_seconds = next_frame_at - now
                if wait_seconds > 0:
                    with self._condition:
                        self._condition.wait(
                            timeout=min(wait_seconds, max(0.0, stream_deadline - now))
                        )
                    continue

                with self._condition:
                    source_frame = self._frame
                    source_sequence = self._frame_sequence
                if source_frame is not None and source_sequence != last_delivered_sequence:
                    last_frame = source_frame
                    last_delivered_sequence = source_sequence
                    if on_source_frame is not None:
                        on_source_frame(source_frame, source_sequence)
                    else:
                        on_frame(source_frame)
                elif on_no_new_frame is not None:
                    on_no_new_frame()

                now = time.monotonic()
                if now - last_renewed_at >= 0.5:
                    self._publish_command("renew", client_id, lease_seconds)
                    last_renewed_at = now
                next_frame_at += frame_interval
                if next_frame_at <= now:
                    # Encoding/network backpressure may make us late. Drop old
                    # time slots instead of sending a burst of stale frames.
                    next_frame_at = now + frame_interval

            # A frame may arrive after the final pacing slot. Return that
            # newest complete source frame for photo/vision consumers without
            # sending it outside the requested stream cadence.
            with self._condition:
                if self._frame is not None and self._frame_sequence > last_delivered_sequence:
                    last_frame = self._frame
            return last_frame
        finally:
            self._publish_command("release", client_id)


def save_camera_photo(jpeg: bytes, directory: str | Path) -> Path:
    """Atomically save one complete source JPEG and return its absolute path."""
    data = bytes(jpeg or b"")
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("camera photo is not a complete JPEG")
    target_directory = Path(directory).expanduser().resolve()
    target_directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = target_directory / f"wali_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    temporary = target.with_suffix(".jpg.tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target
