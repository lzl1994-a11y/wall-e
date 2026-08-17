#!/usr/bin/env python3
"""Isolated ROS/UVC worker for the configuration-page camera preview."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from typing import Any


ROS_IMAGE_TOPICS = ("/image_padded_jpeg", "/image")


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def jpeg_from_ros_message(message: Any) -> bytes | None:
    """Return JPEG bytes from sensor_msgs/Image or CompressedImage."""
    encoding = str(getattr(message, "encoding", "")).lower()
    compressed_format = str(getattr(message, "format", "")).lower()
    raw = bytes(getattr(message, "data", b""))
    if (
        encoding in {"jpeg", "jpg", "mjpeg"}
        or "jpeg" in compressed_format
        or "jpg" in compressed_format
        or raw.startswith(b"\xff\xd8")
    ):
        return raw or None

    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    channels = 1 if encoding in {"mono8", "8uc1"} else 3
    height = int(getattr(message, "height", 0))
    width = int(getattr(message, "width", 0))
    if height <= 0 or width <= 0:
        return None
    array = np.frombuffer(raw, dtype=np.uint8)
    expected = height * width * channels
    if array.size < expected:
        return None
    array = array[:expected].reshape((height, width, channels))
    if channels == 1:
        image = array
    elif encoding.startswith("rgb"):
        image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    else:
        image = array
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else None


def stream_ros_frames(wait_seconds: float, frame_rate: float) -> tuple[bool, str]:
    """Reuse the active ROS vision pipeline and explain why a UVC fallback is needed."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image
    except ImportError as exc:
        return False, f"ROS Python 环境不可用: {exc}"

    initialized_here = False
    node = None
    try:
        rclpy.init(args=[])
        initialized_here = True
        node = Node(f"wali_camera_preview_{int(time.time() * 1000) % 1000000}")
        first_frame = False
        last_emit_at = 0.0
        padded_seen_at = 0.0
        frame_interval = 1.0 / max(1.0, frame_rate)
        sample_started = time.monotonic()
        sample_frames = 0
        measured_fps = 0.0

        def on_image(message: Any, source: str) -> None:
            nonlocal first_frame, last_emit_at, padded_seen_at
            nonlocal sample_started, sample_frames, measured_fps
            now = time.monotonic()
            if source == "/image_padded_jpeg":
                padded_seen_at = now
            elif now - padded_seen_at < 1.0:
                return
            if now - last_emit_at < frame_interval:
                return
            jpeg = jpeg_from_ros_message(message)
            if not jpeg:
                return

            first_frame = True
            last_emit_at = now
            sample_frames += 1
            elapsed = now - sample_started
            if elapsed >= 1.0:
                measured_fps = sample_frames / elapsed
                sample_started = now
                sample_frames = 0
            emit({
                "type": "frame",
                "jpeg": base64.b64encode(jpeg).decode("ascii"),
                "width": int(getattr(message, "width", 0)),
                "height": int(getattr(message, "height", 0)),
                "fps": measured_fps,
                "source": source,
            })

        subscriptions = []
        for topic in ROS_IMAGE_TOPICS:
            subscriptions.append(node.create_subscription(
                Image,
                topic,
                lambda message, source=topic: on_image(message, source),
                qos_profile_sensor_data,
            ))
            # Horizon camera/codec releases differ: JPEG topics may be either
            # Image(encoding=jpeg) or CompressedImage(format=jpeg). DDS only
            # connects the subscription with the matching topic type.
            subscriptions.append(node.create_subscription(
                CompressedImage,
                topic,
                lambda message, source=topic: on_image(message, source),
                qos_profile_sensor_data,
            ))
        emit({"type": "status", "phase": "waiting_ros", "source": "ROS image topic"})

        deadline = time.monotonic() + max(0.1, wait_seconds)
        while rclpy.ok() and not first_frame and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not first_frame:
            return (
                False,
                f"{wait_seconds:g} 秒内未收到 /image_padded_jpeg 或 /image 的可用画面",
            )

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
        return True, ""
    except BrokenPipeError:
        return True, ""
    except Exception as exc:
        return False, f"订阅 ROS 摄像头话题失败: {exc}"
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if initialized_here:
            try:
                rclpy.shutdown()
            except Exception:
                pass


def stream_uvc_frames(device: str, frame_rate: float) -> int:
    try:
        import cv2
    except ImportError:
        emit({"type": "error", "error": "当前 Python 环境未安装 OpenCV，无法直连摄像头"})
        return 2

    capture = None
    try:
        emit({"type": "status", "phase": "opening", "source": device})
        backend = getattr(cv2, "CAP_V4L2", None) if device.startswith("/dev/video") else None
        capture = cv2.VideoCapture(device, backend) if backend is not None else cv2.VideoCapture(device)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {device}，设备可能正被其他进程占用")

        fourcc_property = getattr(cv2, "CAP_PROP_FOURCC", None)
        fourcc_factory = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_property is not None and fourcc_factory is not None:
            capture.set(fourcc_property, fourcc_factory(*"MJPG"))
        for prop_name, value in (
            ("CAP_PROP_FRAME_WIDTH", 640),
            ("CAP_PROP_FRAME_HEIGHT", 480),
            ("CAP_PROP_FPS", frame_rate),
            ("CAP_PROP_BUFFERSIZE", 1),
        ):
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                capture.set(prop, value)

        emit({"type": "status", "phase": "waiting_frame", "source": device})
        frame_interval = 1.0 / max(1.0, frame_rate)
        sample_started = time.monotonic()
        sample_frames = 0
        measured_fps = 0.0
        failed_reads = 0
        while True:
            frame_started = time.monotonic()
            ok, image = capture.read()
            if not ok or image is None:
                failed_reads += 1
                if failed_reads >= 20:
                    raise RuntimeError("摄像头连续读帧失败")
                time.sleep(0.05)
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
            emit({
                "type": "frame",
                "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(width),
                "height": int(height),
                "fps": measured_fps,
                "source": device,
            })

            remaining = frame_interval - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    except BrokenPipeError:
        return 0
    except Exception as exc:
        try:
            emit({"type": "error", "error": str(exc)})
        except BrokenPipeError:
            pass
        return 1
    finally:
        if capture is not None:
            capture.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--ros-wait", type=float, default=4.5)
    args = parser.parse_args()

    using_ros, diagnostic = stream_ros_frames(args.ros_wait, args.fps)
    if using_ros:
        return 0
    emit({
        "type": "status",
        "phase": "ros_fallback",
        "source": "UVC 直连回退",
        "diagnostic": diagnostic,
    })
    return stream_uvc_frames(args.device, args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
