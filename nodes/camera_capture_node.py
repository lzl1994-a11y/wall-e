#!/usr/bin/env python3
"""Single lifecycle owner for a hot-standby camera stream."""

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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.vision.camera_capture_protocol import (
    CAMERA_CAPTURE_SERVICE,
    CAMERA_FIRST_FRAME_TIMEOUT_SEC,
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_SOURCE_TOPIC,
    CAMERA_STATUS_TOPIC,
    TRACKING_CAMERA_FRAME_TOPIC,
    CameraLeaseBook,
    CameraWatchdogAction,
    _prepare_hobot_camera_command,
    decode_camera_command,
    evaluate_camera_watchdog,
    jpeg_from_ros_image,
)
from services.hardware.usb_devices import resolve_camera_device
from services.ros_environment import DEFAULT_TROS_SETUP


class CameraCaptureNode(Node):
    RETRY_DELAY_SEC = 1.0
    FRAME_TIMEOUT_SEC = 3.0
    CAPTURE_REQUEST_TIMEOUT_SEC = 5.0
    DECODE_VALIDATION_INTERVAL_SEC = 1.0
    # ros2 run + hobot_usb_cam may spend several seconds enumerating V4L2
    # nodes before it publishes the first ROS image. Keep this watchdog longer
    # than device initialization so a slow first open is not mistaken for a
    # dead camera.
    # On the X3 board the ROS wrapper and UVC control probing can take more
    # than 15 seconds while the rest of the robot starts.  Keep the timeout
    # bounded, but allow the supported camera mode to finish initialization.
    FIRST_FRAME_TIMEOUT_SEC = CAMERA_FIRST_FRAME_TIMEOUT_SEC

    def __init__(self) -> None:
        super().__init__("camera_capture_node")
        self._leases = CameraLeaseBook()
        self._camera_process: subprocess.Popen | None = None
        self._camera_device = ""
        self._camera_size = (640, 480)
        self._reconfiguring = False
        self._last_source_frame = 0.0
        self._last_output_frame = 0.0
        self._last_decode_validation = 0.0
        self._process_started_at = 0.0
        self._retry_after = 0.0
        self._last_status_signature: tuple | None = None
        self._last_status_publish = 0.0
        self._capture_enabled: bool | None = None
        self._capture_request = None
        self._capture_request_target: bool | None = None
        self._capture_request_started_at = 0.0
        self._capture_transition_started_at = 0.0
        self._capture_service_warning_at = 0.0

        # ``hobot_usb_cam`` is launched only here and publishes the canonical
        # JPEG stream on /image.  The deployed TogetherROS hobot_usb_cam
        # publishes that MJPEG as CompressedImage.
        # This node adapts it to preview and lease-gated detection topics,
        # so photo/TFT/web consumers never open
        # the V4L2 device themselves.
        self._frame_pub = self.create_publisher(
            CompressedImage,
            CAMERA_FRAME_TOPIC,
            # hobot_codec uses a Reliable subscription. A Reliable publisher
            # remains compatible with Best Effort preview subscribers, while
            # the reverse pairing drops every detector frame.
            10,
        )
        self._status_pub = self.create_publisher(String, CAMERA_STATUS_TOPIC, 10)
        self._tracking_frame_pub = self.create_publisher(
            CompressedImage, TRACKING_CAMERA_FRAME_TOPIC, 10
        )
        self._capture_client = self.create_client(SetBool, CAMERA_CAPTURE_SERVICE)
        self.create_subscription(String, CAMERA_COMMAND_TOPIC, self._on_command, 10)
        # ROS 2 Humble does not allow one node to subscribe to the same topic
        # with incompatible message types. Keep this endpoint aligned with the
        # actual hobot_usb_cam publisher instead of probing both wire types.
        self.create_subscription(
            CompressedImage,
            CAMERA_SOURCE_TOPIC,
            self._on_source_image,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(0.2, self._tick)
        # Keep the camera process and V4L2 configuration warm for this node's
        # lifetime. Leases toggle VIDIOC_STREAMON/OFF through hobot_usb_cam's
        # set_capture service, so standby does not transfer or generate frames.
        self._tick()
        self.get_logger().info(
            f"摄像头热备节点上线: {CAMERA_SOURCE_TOPIC} -> {CAMERA_FRAME_TOPIC}"
        )

    def _on_command(self, message: String) -> None:
        command = decode_camera_command(message.data)
        if command is None:
            self.get_logger().warn("忽略无效的摄像头请求")
            return
        if command["action"] == "release":
            self._leases.release(command["client_id"])
        else:
            self._leases.acquire(command["client_id"], command["lease_sec"], resolution=command["resolution"])
        self._tick()
        if command["action"] != "release":
            state, source, error, *_ = self._last_status_signature
            self._publish_status(state, source=source, error=error, force=True, request=command)

    def _on_source_image(self, message: CompressedImage) -> None:
        now = time.monotonic()
        validate_decode = (
            self._last_decode_validation <= 0.0
            or now - self._last_decode_validation
            >= self.DECODE_VALIDATION_INTERVAL_SEC
        )
        self._leases.purge(now=now)
        normalize = self._leases.needs_normalized_frames
        # Hardware-only remote calls consume /image and flip in the encoder.
        # AI/photos/TFT and software WebRTC still receive normalized JPEG.
        jpeg = jpeg_from_ros_image(
            message,
            validate_decode=validate_decode,
            flip_vertical=normalize,
            expected_size=self._camera_size,
        )
        if not jpeg:
            return
        # A running process alone is not healthy: only a complete, decodable
        # JPEG counts as a source frame for the first-frame watchdog.
        self._last_source_frame = now
        self._reconfiguring = False
        if validate_decode:
            self._last_decode_validation = now
        if not self._leases.active:
            state = "standby" if self._capture_enabled is False else "stopping"
            self._publish_status(state, source=CAMERA_SOURCE_TOPIC)
            return
        if normalize:
            frame = CompressedImage(
                header=message.header,
                format="jpeg",
                data=jpeg,
            )
            self._frame_pub.publish(frame)
            # Preview leases must not feed the hot-standby detector. Reuse
            # tracking's existing expiring lease instead of another gate state.
            if self._leases.tracking_active:
                self._tracking_frame_pub.publish(frame)
        self._last_output_frame = time.monotonic()
        self._publish_status("streaming", source=CAMERA_SOURCE_TOPIC)

    def _tick(self) -> None:
        now = time.monotonic()
        self._leases.purge(now=now)
        process = self._camera_process
        if (
            process is not None
            and process.poll() is None
            and self._leases.active
            and self._camera_size != self._leases.resolution
        ):
            # The driver applies dimensions at startup. Keep one device owner;
            # only restart it for an active client's size requirement. With no
            # clients, stop UVC capture and retain the last configured size so
            # the next preview can resume without another V4L2 initialization.
            self._reconfiguring = True
            self._publish_status("starting", source=CAMERA_SOURCE_TOPIC, force=True)
            self._stop_camera_process()
            self._start_camera_process()
            return
        if process is not None and process.poll() is not None:
            code = process.returncode
            self._camera_process = None
            self._camera_device = ""
            self._process_started_at = 0.0
            self._reset_capture_state()
            self._retry_after = now + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_SOURCE_TOPIC,
                error=f"hobot_usb_cam 已退出，退出码 {code}",
                force=True,
            )

        if self._camera_process is not None:
            self._sync_capture_state(now=now)
            if self._camera_process is None:
                return
            if not self._leases.active:
                state = (
                    "standby"
                    if self._capture_enabled is False and self._capture_request is None
                    else "stopping"
                )
                self._publish_status(state, source=CAMERA_SOURCE_TOPIC)
                return
            if self._capture_enabled is not True or self._capture_request is not None:
                self._publish_status("starting", source=CAMERA_SOURCE_TOPIC)
                return

        decision = evaluate_camera_watchdog(
            process_alive=self._camera_process is not None,
            now=now,
            process_started_at=self._process_started_at,
            last_source_frame=self._last_source_frame,
            last_output_frame=self._last_output_frame,
            has_active_leases=self._leases.active,
            retry_after=self._retry_after,
            first_frame_timeout_sec=self.FIRST_FRAME_TIMEOUT_SEC,
            frame_timeout_sec=self.FRAME_TIMEOUT_SEC,
        )

        if decision.action == CameraWatchdogAction.FIRST_FRAME_TIMEOUT:
            device = self._camera_device or "未知设备"
            topic_diagnostic = self._camera_topic_diagnostic()
            self._stop_camera_process()
            self._retry_after = now + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_SOURCE_TOPIC,
                error=(
                    f"hobot_usb_cam 首帧等待超时（设备 {device}，已等待 {decision.elapsed_sec:.1f}s）"
                    f"{('；' + topic_diagnostic) if topic_diagnostic else ''}"
                ),
                force=True,
            )
            return

        if decision.action == CameraWatchdogAction.FRAME_TIMEOUT:
            device = self._camera_device or "未知设备"
            self._stop_camera_process()
            self._retry_after = now + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_SOURCE_TOPIC,
                error=(
                    f"hobot_usb_cam 画面中断（设备 {device}，"
                    f"{decision.elapsed_sec:.1f}s 没有有效新帧）"
                ),
                force=True,
            )
            return

        if decision.action == CameraWatchdogAction.START_PROCESS:
            self._start_camera_process()
            return

        if decision.action == CameraWatchdogAction.PUBLISH_STATUS:
            self._publish_status(decision.state, source=CAMERA_SOURCE_TOPIC)

    def _sync_capture_state(self, *, now: float | None = None) -> None:
        """Drive hobot_usb_cam's V4L2 stream without restarting its process."""
        if self._camera_process is None:
            return
        current_time = time.monotonic() if now is None else now
        if self._capture_request is not None:
            if current_time - self._capture_request_started_at < self.CAPTURE_REQUEST_TIMEOUT_SEC:
                return
            # A request sent while the driver restarts can lose its response.
            # Drop that future and reconcile against the latest lease state.
            waiting_since = self._capture_transition_started_at
            self._reset_capture_state()
            if current_time - waiting_since >= self.FIRST_FRAME_TIMEOUT_SEC:
                self._stop_camera_process()
                self._retry_after = current_time + self.RETRY_DELAY_SEC
                self._publish_status("error", source=CAMERA_SOURCE_TOPIC,
                                     error="摄像头采集服务持续无响应，已停止驱动等待恢复", force=True)
                return
            self._capture_transition_started_at = waiting_since
            self.get_logger().warn("摄像头采集服务响应超时，清理旧请求并重新同步采集状态")
        desired = self._leases.active
        if self._capture_enabled is desired:
            return
        if not self._capture_client.service_is_ready():
            if current_time - self._capture_service_warning_at >= 10.0:
                self._capture_service_warning_at = current_time
                self.get_logger().warn(
                    f"等待 {CAMERA_CAPTURE_SERVICE} 服务，摄像头尚未进入真实热备"
                )
            return

        request = SetBool.Request()
        request.data = desired
        try:
            future = self._capture_client.call_async(request)
        except Exception as exc:
            self.get_logger().warn(f"切换摄像头采集状态失败: {exc}")
            return
        self._capture_request = future
        self._capture_request_target = desired
        self._capture_request_started_at = current_time
        if not self._capture_transition_started_at:
            self._capture_transition_started_at = current_time
        future.add_done_callback(self._on_capture_response)

    def _on_capture_response(self, future) -> None:
        if future is not self._capture_request:
            return
        target = self._capture_request_target
        self._capture_request = None
        self._capture_request_target = None
        self._capture_request_started_at = 0.0
        self._capture_transition_started_at = 0.0
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"摄像头采集服务调用失败: {exc}")
            return

        # The deployed hobot_usb_cam executes STREAMON/OFF and returns the
        # message below, but leaves SetBool.success at its default false value.
        # Accept its explicit completion message while still honoring corrected
        # package versions that set success=true.
        expected = "Start Capturing" if target else "Stop Capturing"
        if not getattr(response, "success", False) and getattr(response, "message", "") != expected:
            self.get_logger().warn(
                f"摄像头采集服务拒绝请求: {getattr(response, 'message', '')}"
            )
            return

        self._capture_enabled = bool(target)
        now = time.monotonic()
        if self._capture_enabled:
            self._last_source_frame = 0.0
            self._last_output_frame = 0.0
            self._last_decode_validation = 0.0
            self._process_started_at = now
            self._publish_status("starting", source=CAMERA_SOURCE_TOPIC, force=True)
            self.get_logger().info("摄像头退出热备，恢复 UVC 取流")
        else:
            self._last_output_frame = 0.0
            self._publish_status("standby", source=CAMERA_SOURCE_TOPIC, force=True)
            self.get_logger().info("摄像头进入真实热备，UVC 已停止取流")

        # A lease may have changed while the service request was in flight.
        self._sync_capture_state(now=now)

    def _reset_capture_state(self) -> None:
        pending = self._capture_request
        self._capture_enabled = None
        self._capture_request = None
        self._capture_request_target = None
        self._capture_request_started_at = 0.0
        self._capture_transition_started_at = 0.0
        if pending is not None:
            self._capture_client.remove_pending_request(pending)
            pending.cancel()

    def _camera_topic_diagnostic(self) -> str:
        """Report graph state without assuming the camera message type."""
        try:
            graph = dict(self.get_topic_names_and_types())
        except Exception:
            return "ROS 图谱查询失败"
        details = []
        for topic in (CAMERA_SOURCE_TOPIC, CAMERA_FRAME_TOPIC):
            types = ",".join(graph.get(topic, [])) or "无类型"
            try:
                publishers = self.count_publishers(topic)
            except Exception:
                publishers = "?"
            details.append(f"{topic}: {types}, publishers={publishers}")
        return "ROS 图谱 " + "；".join(details)

    def _start_camera_process(self) -> None:
        device = resolve_camera_device()
        if not device:
            self._retry_after = time.monotonic() + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_SOURCE_TOPIC,
                error="未找到已配置的摄像头设备",
                force=True,
            )
            return

        # Use the same snapshot for the reuse decision and the child process.
        # An explicit camera setup override continues to use the old path.
        camera_env = os.environ.copy()
        ros_setup = camera_env.get("WALI_CAMERA_ROS_SETUP", DEFAULT_TROS_SETUP)
        self._camera_size = self._leases.resolution
        command, reused_tros = _prepare_hobot_camera_command(
            device,
            ros_setup=ros_setup,
            resolution=self._camera_size,
            environment=camera_env,
        )
        try:
            self._camera_process = subprocess.Popen(
                command,
                env=camera_env,
                start_new_session=(os.name != "nt"),
            )
        except Exception as exc:
            self._retry_after = time.monotonic() + self.RETRY_DELAY_SEC
            self._publish_status(
                "error",
                source=CAMERA_SOURCE_TOPIC,
                error=f"启动 hobot_usb_cam 失败: {exc}",
                force=True,
            )
            return
        if reused_tros:
            self.get_logger().info("相机进程复用已加载的 TROS 环境")
        self._camera_device = str(device)
        self._last_source_frame = 0.0
        self._last_output_frame = 0.0
        self._last_decode_validation = 0.0
        self._process_started_at = time.monotonic()
        # hobot_usb_cam starts V4L2 capture during process initialization.
        self._capture_enabled = True
        self._capture_request = None
        self._capture_request_target = None
        self._capture_request_started_at = 0.0
        self._capture_transition_started_at = 0.0
        self._publish_status("starting", source=CAMERA_SOURCE_TOPIC, force=True)
        self.get_logger().info(
            f"启动唯一 hobot_usb_cam: {device} -> {CAMERA_SOURCE_TOPIC}"
        )

    def _stop_camera_process(self) -> None:
        process = self._camera_process
        self._camera_process = None
        self._reset_capture_state()
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
        request: dict | None = None,
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
            "reconfiguring": self._reconfiguring,
            "tracking": self._leases.tracking_active,
            "high_resolution": self._leases.resolution == (1280, 720),
            "width": self._camera_size[0],
            "height": self._camera_size[1],
        }
        if request is not None:
            payload["request"] = {
                "client_id": request["client_id"],
                "width": request["resolution"][0],
                "height": request["resolution"][1],
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
