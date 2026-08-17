"""提供一次性的最新摄像头 JPEG 帧。

优先复用视觉管线发布的 ROS Image，避免重复打开摄像头；如果视觉跟随
没有启动，则按 usb_devices/vision.camera_index 临时打开 UVC 设备抓一帧。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - optional on non-robot development hosts
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sensor_msgs.msg import CompressedImage, Image
except ImportError:  # pragma: no cover - ROS is unavailable in unit tests
    Image = None
    CompressedImage = None

try:
    from rclpy.qos import qos_profile_sensor_data
except ImportError:  # pragma: no cover
    qos_profile_sensor_data = 10

from services.usb_devices import resolve_camera_device


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
        "what is this", "what do you see",
    )
    return any(marker in text for marker in markers)


class CameraFrameProvider:
    """缓存 ROS 图像并提供线程安全的 ``capture`` 操作。"""

    def __init__(self, node: Any, config_path: str | Path = "core/config.yaml"):
        self._node = node
        self._config_path = config_path
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[bytes, float]] = {}
        self._subscriptions = []
        self._subscription_keys: set[tuple[str, str]] = set()
        self._subscription_timer = None

        self._ensure_subscriptions()
        try:
            self._subscription_timer = node.create_timer(1.0, self._ensure_subscriptions)
        except Exception as exc:
            node.get_logger().debug(f"camera topic discovery timer unavailable: {exc}")

    def _ensure_subscriptions(self) -> None:
        """Subscribe only to the concrete image type advertised by the ROS graph."""
        message_types = {
            "sensor_msgs/msg/Image": Image,
            "sensor_msgs/msg/CompressedImage": CompressedImage,
        }
        try:
            graph = self._node.get_topic_names_and_types()
        except Exception as exc:
            self._node.get_logger().debug(f"camera topic discovery failed: {exc}")
            return
        if not isinstance(graph, (list, tuple)):
            return

        discovered = dict(graph)
        for topic in ("/image_padded_jpeg", "/image"):
            for type_name in discovered.get(topic, []):
                message_type = message_types.get(type_name)
                key = (topic, type_name)
                if message_type is None or key in self._subscription_keys:
                    continue
                try:
                    subscription = self._node.create_subscription(
                        message_type,
                        topic,
                        lambda msg, source=topic: self._on_image(msg, source),
                        qos_profile_sensor_data,
                    )
                except Exception as exc:
                    self._node.get_logger().debug(
                        f"camera topic unavailable ({topic}, {type_name}): {exc}"
                    )
                    continue
                self._subscriptions.append(subscription)
                self._subscription_keys.add(key)

    def _on_image(self, msg: Any, source: str = "/image") -> None:
        try:
            encoding = str(getattr(msg, "encoding", "")).lower()
            compressed_format = str(getattr(msg, "format", "")).lower()
            raw = bytes(msg.data)
            if (
                encoding in {"jpeg", "jpg", "mjpeg"}
                or "jpeg" in compressed_format
                or "jpg" in compressed_format
                or raw.startswith(b"\xff\xd8")
            ):
                jpeg = raw
            elif cv2 is not None and np is not None:
                channels = 1 if encoding in {"mono8", "8uc1"} else 3
                arr = np.frombuffer(raw, dtype=np.uint8)
                expected = int(msg.height) * int(msg.width) * channels
                if arr.size < expected:
                    return
                arr = arr[:expected].reshape((int(msg.height), int(msg.width), channels))
                if channels == 1:
                    image = arr
                elif encoding.startswith("rgb"):
                    image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    image = arr
                ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    return
                jpeg = encoded.tobytes()
            else:
                return
            if jpeg:
                with self._lock:
                    self._frames[source] = (jpeg, time.monotonic())
        except Exception:
            return

    def _cached(self, max_age: float = 3.0) -> bytes | None:
        with self._lock:
            now = time.monotonic()
            # The padded topic has already applied the robot camera's required
            # horizontal/vertical correction.  Raw /image is only a fallback.
            for source in ("/image_padded_jpeg", "/image"):
                item = self._frames.get(source)
                if item and now - item[1] <= max_age:
                    return item[0]
            return None

    def _capture_uvc(self) -> bytes | None:
        if cv2 is None:
            return None
        device = resolve_camera_device(self._config_path)
        if not device:
            return None
        cap = cv2.VideoCapture(device)
        try:
            if not cap.isOpened():
                return None
            # Discard one buffered frame so the answer reflects the current view.
            cap.grab()
            ok, frame = cap.read()
            if not ok:
                return None
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return encoded.tobytes() if ok else None
        finally:
            cap.release()

    def capture(self, timeout: float = 1.5) -> bytes | None:
        """返回最新 JPEG；ROS 无帧时回退到 UVC 单帧采集。"""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            jpeg = self._cached()
            if jpeg:
                return jpeg
            time.sleep(0.03)
        return self._capture_uvc()
