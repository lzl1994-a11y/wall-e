#!/usr/bin/env python3
"""Single lifecycle owner for on-demand camera frames."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_STATUS_TOPIC,
    TRACKING_IMAGE_TOPIC,
    CameraLeaseBook,
    build_hobot_camera_command,
    decode_camera_command,
)
from services.usb_devices import resolve_camera_device


class CameraCaptureNode(Node):
    SOURCE_FRESH_SEC = 0.8
    RETRY_DELAY_SEC = 1.0
    # ros2 run + hobot_usb_cam may spend several seconds enumerating V4L2
    # nodes before it publishes the first ROS image. Keep this watchdog longer
    # than device initialization so a slow first open is not mistaken for a
    # dead camera.
    FIRST_FRAME_TIMEOUT_SEC = 15.0

    def __init__(self) -> None:
        super().__init__("camera_capture_node")
        self._leases = CameraLeaseBook()
        self._camera_process: subprocess.Popen | None = None
        self._camera_device = ""
        self._last_source_frame = 0.0
        self._last_output_frame = 0.0
        self._process_started_at = 0.0
        self._retry_after = 0.0
        self._last_status_signature: tuple | None = None
        self._last_status_publish = 0.0

        self._frame_pub = self.create_publisher(Image, CAMERA_FRAME_TOPIC, qos_profile_sensor_data)
        self._status_pub = self.create_publisher(String, CAMERA_STATUS_TOPIC, 10)
        self.create_subscription(String, CAMERA_COMMAND_TOPIC, self._on_command, 10)
        self.create_subscription(Image, TRACKING_IMAGE_TOPIC, self._on_tracking_image, qos_profile_sensor_data)
        self.create_subscription(Image, CAMERA_FRAME_TOPIC, self._on_camera_frame, qos_profile_sensor_data)
        self._timer = self.create_timer(0.2, self._tick)
        self._publish_status("idle", source="", force=True)
        self.get_logger().info(
            f"按需摄像头节点上线: {CAMERA_COMMAND_TOPIC} -> {CAMERA_FRAME_TOPIC}"
        )

    def _on_command(self, message: String) -> None:
        command = decode_camera_command(message.data)
        if command is None:
            self.get_logger().warn("忽略无效的摄像头请求")
            return
        if command["action"] == "release":
            self._leases.release(command["client_id"])
        else:
            self._leases.acquire(command["client_id"], command["lease_sec"])
        self._tick()

    def _on_tracking_image(self, message: Image) -> None:
        self._last_source_frame = time.monotonic()
        if not self._leases.active:
            return
        if self._camera_process is not None:
            self._stop_camera_process()
        self._frame_pub.publish(message)
        self._publish_status("streaming", source=TRACKING_IMAGE_TOPIC)

    def _on_camera_frame(self, _message: Image) -> None:
        self._last_output_frame = time.monotonic()
        if self._leases.active and not self._tracking_source_fresh():
            self._publish_status("streaming", source=CAMERA_FRAME_TOPIC)

    def _tracking_source_fresh(self) -> bool:
        return time.monotonic() - self._last_source_frame <= self.SOURCE_FRESH_SEC

    def _tracking_publisher_present(self) -> bool:
        try:
            return self.count_publishers(TRACKING_IMAGE_TOPIC) > 0
        except Exception:
            return False

    def _tick(self) -> None:
        now = time.monotonic()
        self._leases.purge(now=now)
        process = self._camera_process
        if process is not None and process.poll() is not None:
            code = process.returncode
            self._camera_process = None
            self._camera_device = ""
            self._process_started_at = 0.0
            self._retry_after = now + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_FRAME_TOPIC,
                error=f"hobot_usb_cam 已退出，退出码 {code}",
                force=True,
            )

        if not self._leases.active:
            self._stop_camera_process()
            self._publish_status("idle", source="")
            return

        tracking_fresh = self._tracking_source_fresh()
        if tracking_fresh or self._tracking_publisher_present():
            self._stop_camera_process()
            self._publish_status(
                "streaming" if tracking_fresh else "starting",
                source=TRACKING_IMAGE_TOPIC,
            )
            return

        if (
            self._camera_process is not None
            and self._last_output_frame < self._process_started_at
            and now - self._process_started_at > self.FIRST_FRAME_TIMEOUT_SEC
        ):
            waited = now - self._process_started_at
            device = self._camera_device or "未知设备"
            self._stop_camera_process()
            self._retry_after = now + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_FRAME_TOPIC,
                error=f"hobot_usb_cam 首帧等待超时（设备 {device}，已等待 {waited:.1f}s）",
                force=True,
            )
            return

        if self._camera_process is None and now >= self._retry_after:
            self._start_camera_process()
            return

        if self._camera_process is not None:
            state = "streaming" if now - self._last_output_frame <= 1.0 else "starting"
            self._publish_status(state, source=CAMERA_FRAME_TOPIC)

    def _start_camera_process(self) -> None:
        device = resolve_camera_device()
        if not device:
            self._retry_after = time.monotonic() + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_FRAME_TOPIC,
                error="未找到已配置的摄像头设备",
                force=True,
            )
            return

        ros_setup = os.environ.get("WALI_CAMERA_ROS_SETUP", "/opt/tros/humble/setup.bash")
        command = build_hobot_camera_command(device, ros_setup=ros_setup)
        try:
            self._camera_process = subprocess.Popen(
                command,
                env=os.environ.copy(),
                start_new_session=(os.name != "nt"),
            )
        except Exception as exc:
            self._retry_after = time.monotonic() + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_FRAME_TOPIC,
                error=f"启动 hobot_usb_cam 失败: {exc}",
                force=True,
            )
            return
        self._camera_device = str(device)
        self._last_output_frame = 0.0
        self._process_started_at = time.monotonic()
        self._publish_status("starting", source=CAMERA_FRAME_TOPIC, force=True)
        self.get_logger().info(
            f"按需启动 hobot_usb_cam: {device} -> {CAMERA_FRAME_TOPIC}"
        )

    def _stop_camera_process(self) -> None:
        process = self._camera_process
        self._camera_process = None
        if process is None or process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            else:
                process.terminate()
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
        self._camera_device = ""
        self._process_started_at = 0.0

    def _publish_status(
        self,
        state: str,
        *,
        source: str,
        error: str = "",
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        signature = (state, source, error, self._leases.count, self._camera_device)
        if not force and signature == self._last_status_signature and now - self._last_status_publish < 2.0:
            return
        payload = {
            "state": state,
            "source": source,
            "device": self._camera_device,
            "clients": self._leases.count,
            "error": error,
        }
        self._status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )
        self._last_status_signature = signature
        self._last_status_publish = now

    def destroy_node(self) -> None:
        self._stop_camera_process()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
