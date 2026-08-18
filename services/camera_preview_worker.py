#!/usr/bin/env python3
"""ROS worker that leases and streams the dedicated /camera_frame topic."""

from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
    jpeg_from_ros_image,
)


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def stream_camera_frames(frame_rate: float, *, lease_sec: float = 4.0) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
    except ImportError as exc:
        emit({"type": "error", "error": f"ROS 摄像头环境不可用: {exc}"})
        return 2

    initialized_here = False
    node = None
    stopping = False
    client_id = f"web-{uuid.uuid4().hex}"
    command_pub = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    try:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        rclpy.init(args=[])
        initialized_here = True
        node = Node(f"wali_camera_preview_{int(time.time() * 1000) % 1000000}")
        command_pub = node.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        frame_interval = 1.0 / max(1.0, frame_rate)
        last_emit_at = 0.0
        sample_started = time.monotonic()
        sample_frames = 0
        measured_fps = 0.0
        manager_acknowledged = False

        def publish_command(action: str) -> None:
            if command_pub is None:
                return
            command_pub.publish(
                String(data=encode_camera_command(action, client_id, lease_sec))
            )

        def on_status(message: String) -> None:
            nonlocal manager_acknowledged
            try:
                status = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                return
            if not isinstance(status, dict):
                return
            manager_state = str(status.get("state", ""))
            if manager_state in {"starting", "streaming"}:
                manager_acknowledged = True
            phase = "waiting_frame" if manager_acknowledged else "requesting_camera"
            emit({
                "type": "status",
                "phase": phase,
                "source": CAMERA_FRAME_TOPIC,
                "diagnostic": str(status.get("error", "")),
            })

        def on_image(message: CompressedImage) -> None:
            nonlocal last_emit_at, sample_started, sample_frames, measured_fps
            now = time.monotonic()
            if now - last_emit_at < frame_interval:
                return
            jpeg = jpeg_from_ros_image(message, quality=82)
            if not jpeg:
                return
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
                "source": CAMERA_FRAME_TOPIC,
            })

        subscriptions = [
            node.create_subscription(
                CompressedImage,
                CAMERA_FRAME_TOPIC,
                on_image,
                qos_profile_sensor_data,
            ),
            node.create_subscription(String, CAMERA_STATUS_TOPIC, on_status, 10),
        ]
        emit({
            "type": "status",
            "phase": "requesting_camera",
            "source": CAMERA_FRAME_TOPIC,
        })

        next_renew = 0.0
        while rclpy.ok() and not stopping:
            now = time.monotonic()
            if now >= next_renew:
                publish_command("acquire" if next_renew == 0.0 else "renew")
                renew_interval = max(0.5, lease_sec / 2.0) if last_emit_at else 0.5
                next_renew = now + renew_interval
            rclpy.spin_once(node, timeout_sec=0.2)
        return 0
    except BrokenPipeError:
        return 0
    except Exception as exc:
        try:
            emit({"type": "error", "error": f"读取 {CAMERA_FRAME_TOPIC} 失败: {exc}"})
        except BrokenPipeError:
            pass
        return 1
    finally:
        if command_pub is not None:
            try:
                command_pub.publish(
                    String(data=encode_camera_command("release", client_id))
                )
                if node is not None and initialized_here and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.1)
            except Exception:
                pass
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
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()
    return stream_camera_frames(args.fps)


if __name__ == "__main__":
    raise SystemExit(main())
