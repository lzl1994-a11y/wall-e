#!/usr/bin/env python3
"""
瓦力视觉跟踪中枢 (深度物理结构适配版)
特点：
1. 摄像头无法左右转动，完全依赖底盘差速跟随。
2. 第5个舵机 (head_yaw) 作为“生动仿生头”，虚假转头以增强生命感。
3. 第6、7个舵机 (neck_top, neck_bottom) 联合控制仰俯。
"""

import time
import json
import signal
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String, Int32
from services.action.action_command import ACTION_COMMAND_TOPIC, parse_action_request
from services.action.action_status import ACTION_STATUS_TOPIC, build_action_status
from services.motion.motion_arbiter import MOTOR_TRACKING_TOPIC
from services.motion.servo_motion_config import load_neck_kinematics
from services.motion.tracking_control import (
    LossState,
    TrackingController,
    TrackingExitReason,
)

from services.vision.vision_pipeline_protocol import (
    TRACKING_SERVO_TARGET_TOPIC,
    VISION_PIPELINE_COMMAND_TOPIC,
    VISION_PIPELINE_START,
    VISION_PIPELINE_STOP,
)
from services.vision.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
)

try:
    from ai_msgs.msg import PerceptionTargets
    HAS_HOBOT_MSGS = True
except ImportError:
    HAS_HOBOT_MSGS = False


class WaliTrackingNode(Node):
    MODE_IDLE = "idle"
    MODE_BODY_FOLLOW = "follow_me"
    MODE_FACE_FOLLOW = "look_at_me"
    MODE_ALIASES = {
        "idle": MODE_IDLE,
        "body_follow": MODE_BODY_FOLLOW,
        "follow_me": MODE_BODY_FOLLOW,
        "face_follow": MODE_FACE_FOLLOW,
        "look_at_me": MODE_FACE_FOLLOW,
    }

    IMG_WIDTH = 960
    IMG_HEIGHT = 544
    BODY_TARGET_RATIO = 0.35  # 跟随模式下的期望身体面积占比

    SEARCH_ROTATE_SPEED = 25  # 丢失目标时的原地转圈速度
    SEARCH_START_DELAY_SEC = 1.0
    SEARCH_STOP_DELAY_SEC = 5.0
    TRACKING_SHUTDOWN_DELAY_SEC = 60.0
    PIPELINE_STARTUP_TIMEOUT_SEC = 180.0
    CAMERA_CLIENT_ID = "tracking-vision"
    CAMERA_LEASE_SEC = 30.0
    CAMERA_RENEW_SEC = 10.0
    DETECTION_WARNING_INTERVAL_SEC = 5.0
    GAZE_START_PITCH = 0.18
    GAZE_MIN_PITCH = -0.20
    GAZE_MAX_PITCH = 0.65
    PITCH_RATE = 0.35  # normalized pitch per second, independent of FPS
    TARGET_MEMORY_SEC = 1.5
    FILTER_TIME_SEC = 0.15
    DETECTION_STALE_SEC = 0.5

    def __init__(self):
        super().__init__('wali_tracking_node')

        if not HAS_HOBOT_MSGS:
            raise RuntimeError(
                "ai_msgs is unavailable; start tracking through launch_nodes.py "
                "or source /opt/tros/humble/setup.bash first"
            )

        self.mode = self.MODE_IDLE
        self._last_time = time.monotonic()
        self._joy_override = False # 若未来恢复 joy_override 机制
        self._mode_started_at = time.monotonic()
        self._last_detection_message = 0.0
        self._last_nonempty_detection = 0.0
        self._last_detection_warning = 0.0
        self._tracking_controller = TrackingController(
            image_width=self.IMG_WIDTH,
            image_height=self.IMG_HEIGHT,
            body_target_ratio=self.BODY_TARGET_RATIO,
            target_memory_seconds=self.TARGET_MEMORY_SEC,
            filter_seconds=self.FILTER_TIME_SEC,
            gaze_start_pitch=self.GAZE_START_PITCH,
            gaze_min_pitch=self.GAZE_MIN_PITCH,
            gaze_max_pitch=self.GAZE_MAX_PITCH,
            pitch_rate=self.PITCH_RATE,
            search_rotate_speed=self.SEARCH_ROTATE_SPEED / 100.0,
            search_start_seconds=self.SEARCH_START_DELAY_SEC,
            search_stop_seconds=self.SEARCH_STOP_DELAY_SEC,
            tracking_shutdown_seconds=self.TRACKING_SHUTDOWN_DELAY_SEC,
            pipeline_startup_timeout_seconds=self.PIPELINE_STARTUP_TIMEOUT_SEC,
        )
        self._neck_kinematics = load_neck_kinematics()

        # ── 订阅与发布 ──
        if HAS_HOBOT_MSGS:
            # Perception is a high-rate latest-value stream. Best-effort input
            # is compatible with both reliable and sensor-data publishers,
            # unlike a reliable subscriber paired with a best-effort model.
            self.create_subscription(
                PerceptionTargets,
                '/hobot_mono2d_body_detection',
                self._on_detection,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(String, '/hobot_mono2d_body_detection', lambda x: None, 10)

        self.create_subscription(String, ACTION_COMMAND_TOPIC, self._on_action_cmd, 10)

        # Detection updates are a high-rate latest-value control stream, not
        # high-level actions. Sending them through /action_cmd repeatedly
        # interrupted sequence_ros_node's 50 Hz interpolation.
        self._tracking_servo_pub = self.create_publisher(
            String,
            TRACKING_SERVO_TARGET_TOPIC,
            QoSProfile(depth=1),
        )
        self._action_status_pub = self.create_publisher(String, ACTION_STATUS_TOPIC, 10)
        self._motor_pub = self.create_publisher(String, MOTOR_TRACKING_TOPIC, 10)
        pipeline_qos = QoSProfile(depth=1)
        pipeline_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._vision_pipeline_pub = self.create_publisher(
            String,
            VISION_PIPELINE_COMMAND_TOPIC,
            pipeline_qos,
        )
        self._camera_command_pub = self.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        self.create_subscription(String, CAMERA_STATUS_TOPIC, self._on_camera_status, 10)
        self._last_camera_renew = 0.0
        self._vision_pipeline_started = False

        # 丢失目标的搜索定时器
        self._timer = self.create_timer(0.1, self._control_tick)
        self._last_target_seen = time.monotonic()

        # Tracking starts in IDLE, so the expensive RDK camera/detector pipeline
        # can remain off until a follow/look command arrives.
        self._publish_vision_pipeline_command(VISION_PIPELINE_STOP)

        self.get_logger().info("视觉跟踪节点上线 (双舵机俯仰 + 底盘左右版本)")

    # ===================================================================
    # 核心检测处理
    # ===================================================================
    def _on_detection(self, msg):
        if self.mode == self.MODE_IDLE or self._joy_override:
            return

        now = time.monotonic()
        detector_was_ready = (
            self._last_detection_message > 0.0
            and self._last_detection_message >= self._mode_started_at
        )
        self._last_detection_message = now
        if not detector_was_ready:
            # Target-loss timeout starts when inference actually comes online,
            # not while its native dependencies are still starting/building.
            self._last_target_seen = now
        elapsed = now - self._last_time
        if elapsed > self.DETECTION_STALE_SEC:
            self._tracking_controller.reset_control_history()
        dt = max(0.001, min(elapsed, 0.1))
        self._last_time = now

        body_boxes = []
        face_boxes = []

        for target in msg.targets:
            for roi in target.rois:
                rect = roi.rect
                if rect.width <= 0 or rect.height <= 0:
                    continue
                cx = rect.x_offset + rect.width / 2.0
                cy = rect.y_offset + rect.height / 2.0
                area_ratio = (rect.width * rect.height) / (self.IMG_WIDTH * self.IMG_HEIGHT)
                if roi.type in ["body", "person"]:
                    body_boxes.append((cx, cy, area_ratio))
                elif roi.type in ["face", "head"]:
                    face_boxes.append((cx, cy, area_ratio))

        if body_boxes or face_boxes:
            self._last_nonempty_detection = now

        if not detector_was_ready:
            roi_types = sorted({
                str(roi.type)
                for target in msg.targets
                for roi in target.rois
            })
            self.get_logger().info(
                "视觉检测链路已连通: "
                f"targets={len(msg.targets)} roi_types={roi_types or '-'}"
            )

        if self.mode == self.MODE_BODY_FOLLOW:
            self._handle_body_follow(body_boxes, dt)
        elif self.mode == self.MODE_FACE_FOLLOW:
            self._handle_face_follow(face_boxes, body_boxes, dt)

    def _handle_body_follow(self, body_boxes, dt):
        """模式 1: 纯底盘跟随 (前进后退+左右转)，摄像头仰俯锁定平视"""
        decision = self._tracking_controller.follow_body(
            body_boxes,
            dt=dt,
            now=time.monotonic(),
        )
        self._apply_tracking_decision(decision)


    def _handle_face_follow(self, face_boxes, body_boxes, dt):
        """模式 2: 禅定注视 (底盘静止，双舵机动态俯仰)"""
        decision = self._tracking_controller.gaze_at_face(
            face_boxes,
            body_boxes,
            dt=dt,
            now=time.monotonic(),
        )
        self._apply_tracking_decision(decision)

    def _apply_tracking_decision(self, decision):
        if decision.target_seen:
            self._mark_target_seen()
        if decision.motor is not None:
            self._publish_motor_diff(decision.motor.left, decision.motor.right)
        if decision.head is not None:
            self._publish_head_and_neck(
                decision.head.x_error,
                decision.head.pitch,
            )


    def _control_tick(self):
        """Search briefly, then fail safe and eventually release the camera."""
        if self.mode == self.MODE_IDLE or self._joy_override:
            return

        self._renew_camera_lease()

        now = time.monotonic()
        lost_seconds = now - self._last_target_seen
        detector_ready = (
            self._last_detection_message > 0.0
            and self._last_detection_message >= self._mode_started_at
        )
        decision = self._tracking_controller.handle_target_loss(
            gaze=self.mode == self.MODE_FACE_FOLLOW,
            detector_ready=detector_ready,
            detector_stale=(
                now - self._last_detection_message > self.DETECTION_STALE_SEC
            ),
            lost_seconds=lost_seconds,
            mode_seconds=now - self._mode_started_at,
        )
        if decision.exit_reason == TrackingExitReason.TARGET_LOST:
            self.get_logger().warning(
                "目标丢失超过60秒，退出视觉跟随并关闭跟踪摄像头"
            )
            self._set_tracking_mode(self.MODE_IDLE)
            return
        if decision.exit_reason == TrackingExitReason.PIPELINE_STARTUP_TIMEOUT:
            self.get_logger().warning(
                "视觉检测管线启动超过180秒仍无消息，退出跟踪并释放摄像头"
            )
            self._set_tracking_mode(self.MODE_IDLE)
            return

        self._warn_if_detection_is_missing(lost_seconds)
        self._apply_loss_decision(decision)
        if (
            self.mode == self.MODE_BODY_FOLLOW
            and decision.state == LossState.SEARCH_STOPPED
            and decision.head is not None
        ):
            self.get_logger().warning("目标丢失超过5秒，停止旋转搜索")

    def _apply_loss_decision(self, decision):
        if decision.motor is not None:
            self._publish_motor_diff(decision.motor.left, decision.motor.right)
        if decision.head is not None:
            self._publish_head_and_neck(decision.head.x_error, decision.head.pitch)

    def _warn_if_detection_is_missing(self, lost_seconds):
        if lost_seconds < self.SEARCH_STOP_DELAY_SEC:
            return
        now = time.monotonic()
        if now - self._last_detection_warning < self.DETECTION_WARNING_INTERVAL_SEC:
            return
        self._last_detection_warning = now
        if self._last_detection_message < self._mode_started_at:
            self.get_logger().warning(
                "视觉检测话题尚无消息：请检查 /image_nv12、"
                "/image_padded_nv12 和 /hobot_mono2d_body_detection_raw"
            )
        elif self._last_nonempty_detection < self._mode_started_at:
            self.get_logger().warning(
                "视觉检测消息已到达，但人体/人脸结果持续为空"
            )

    # ===================================================================
    # 执行层
    # ===================================================================
    def _publish_motor_diff(self, left_speed, right_speed):
        cmd = {
            "left": {"action": 1 if left_speed > 0 else (2 if left_speed < 0 else 0), "throttle": int(abs(left_speed) * 100)},
            "right": {"action": 1 if right_speed > 0 else (2 if right_speed < 0 else 0), "throttle": int(abs(right_speed) * 100)}
        }
        msg = String()
        msg.data = json.dumps(cmd)
        self._motor_pub.publish(msg)

    def _publish_motor(self, left_act, right_act, throttle):
        cmd = {
            "left": {"action": left_act, "throttle": throttle},
            "right": {"action": right_act, "throttle": throttle}
        }
        msg = String()
        msg.data = json.dumps(cmd)
        self._motor_pub.publish(msg)

    def _stop_motor(self):
        self._publish_motor(0, 0, 0)

    def _mark_target_seen(self):
        self._last_target_seen = time.monotonic()
        self._tracking_controller.mark_target_seen()

    def _publish_vision_pipeline_command(self, command):
        self._vision_pipeline_pub.publish(String(data=command))

    def _set_vision_pipeline_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._vision_pipeline_started:
            return
        self._publish_vision_pipeline_command(
            VISION_PIPELINE_START if enabled else VISION_PIPELINE_STOP
        )
        self._vision_pipeline_started = enabled

    def _on_camera_status(self, message):
        try:
            status = json.loads(message.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict) or self.mode == self.MODE_IDLE:
            return
        state = str(status.get("state", ""))
        if state == "streaming":
            self._set_vision_pipeline_enabled(True)
        elif state in {"error", "idle"}:
            self._set_vision_pipeline_enabled(False)

    def _publish_camera_lease(self, action):
        self._camera_command_pub.publish(
            String(data=encode_camera_command(
                action, self.CAMERA_CLIENT_ID, self.CAMERA_LEASE_SEC
            ))
        )

    def _renew_camera_lease(self):
        now = time.monotonic()
        if now - self._last_camera_renew >= self.CAMERA_RENEW_SEC:
            self._publish_camera_lease("renew")
            self._last_camera_renew = now

    def _publish_head_and_neck(self, x_error, pitch_val):
        """
        x_error: [-1.0, 1.0] 目标在左侧则为负
        pitch_val: [-1.0, 1.0] 1.0 为最上，-1.0 为最下，0.0 为平视
        """
        targets = {}

        # 1. 仿生扭头 (直接映射误差，无累积)
        targets['head_yaw'] = int(5000 - x_error * 2600)

        # 2. 脖子仰俯双舵机补偿（标定和中心位置来自 config.yaml）
        targets.update(self._neck_kinematics.targets(pitch_val))

        # Use the dedicated latest-value stream. sequence_ros_node keeps the
        # normal interpolation and collision limits without globally
        # interrupting its action state for every detector frame.
        payload = {
            "targets": targets,
            "step_size": 40.0,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._tracking_servo_pub.publish(msg)

    def _set_tracking_mode(self, requested_mode):
        mode_key = str(requested_mode or "").strip()
        mode = self.MODE_ALIASES.get(mode_key, mode_key)

        if mode in (self.MODE_BODY_FOLLOW, self.MODE_FACE_FOLLOW):
            self.mode = mode
            self._tracking_controller.reset(gaze=mode == self.MODE_FACE_FOLLOW)
            self._stop_motor()
            self._publish_head_and_neck(
                0.0, self._tracking_controller.current_neck_pitch
            )
            now = time.monotonic()
            self._mode_started_at = now
            self._last_detection_message = 0.0
            self._last_nonempty_detection = 0.0
            self._last_detection_warning = 0.0
            self._last_time = now
            self._last_target_seen = now
            # The camera manager is the sole V4L2 owner.  Acquire it before
            # starting consumers so the detector can wait for /image instead
            # of racing a second hobot_usb_cam instance.
            self._publish_camera_lease("acquire")
            self._last_camera_renew = now
            self.get_logger().info(f"Entered tracking mode: {mode} (requested: {mode_key})")
            return True
        elif mode == self.MODE_IDLE:
            self.mode = self.MODE_IDLE
            self._stop_motor()
            self._publish_head_and_neck(0.0, 0.0) # 回中
            self._set_vision_pipeline_enabled(False)
            self._publish_camera_lease("release")
            self.get_logger().info("Tracking mode: IDLE")
            return True
        else:
            self.get_logger().warn(f"Unknown tracking mode: {requested_mode}")
            return False

    def _publish_action_status(self, request, status, detail=""):
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if not request_id:
            return
        self._action_status_pub.publish(String(data=build_action_status(
            request_id,
            request.get("name", "unknown"),
            status,
            source="wali_tracking_node",
            detail=detail,
        )))

    # ===================================================================
    # 模式切换监听
    # ===================================================================
    def _on_action_cmd(self, msg):
        self.get_logger().info(f"[TrackingNode] Received action_cmd: {msg.data}")
        request = parse_action_request(msg.data)
        if request is None:
            self.get_logger().error("[TrackingNode] Failed to parse action request")
            return
        name = request["name"]
        args = request["arguments"]

        self.get_logger().info(f"[TrackingNode] Action name: '{name}', args: {args}")

        if name == "stop_all":
            self._set_tracking_mode(self.MODE_IDLE)
            self._publish_action_status(request, "completed")
        elif name == "set_tracking_mode":
            self._publish_action_status(request, "accepted")
            ok = self._set_tracking_mode(args.get("mode", ""))
            self._publish_action_status(
                request,
                "completed" if ok else "rejected",
                "" if ok else "invalid_tracking_mode",
            )
        elif name == "set_vision_gate":
            enabled = args.get("enabled", False)
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
            self._publish_action_status(request, "accepted")
            ok = self._set_tracking_mode(self.MODE_BODY_FOLLOW if enabled else self.MODE_IDLE)
            self._publish_action_status(
                request,
                "completed" if ok else "rejected",
                "" if ok else "vision_gate_update_failed",
            )


def main(args=None):
    # Keep the context alive until fail-safe motor/camera stop commands publish.
    # rclpy's default SIGINT handler shuts it down before finally can run.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    node = WaliTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._set_tracking_mode(node.MODE_IDLE)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
