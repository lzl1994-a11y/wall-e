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
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String, Int32
from services.action_command import parse_action_request
from services.action_status import ACTION_STATUS_TOPIC, build_action_status
from services.motion_arbiter import MOTOR_TRACKING_TOPIC

from services.vision_pipeline_protocol import (
    TRACKING_SERVO_TARGET_TOPIC,
    VISION_PIPELINE_COMMAND_TOPIC,
    VISION_PIPELINE_START,
    VISION_PIPELINE_STOP,
)
from services.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
)

try:
    from ai_msgs.msg import PerceptionTargets
    HAS_HOBOT_MSGS = True
except ImportError:
    HAS_HOBOT_MSGS = False


class PID:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(min(out, self.out_max), self.out_min)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


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
    CAMERA_CLIENT_ID = "tracking-vision"
    CAMERA_LEASE_SEC = 30.0
    CAMERA_RENEW_SEC = 10.0
    DETECTION_WARNING_INTERVAL_SEC = 5.0

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
        self._search_active = False
        self._search_halted = False
        self._mode_started_at = time.monotonic()
        self._last_detection_message = 0.0
        self._last_nonempty_detection = 0.0
        self._last_detection_warning = 0.0

        # ── PID 控制器 ──
        # 底盘水平追踪 (模式1用)
        self._pid_chassis_yaw = PID(kp=0.6, ki=0.0, kd=0.05, out_min=-1.0, out_max=1.0)
        # 底盘前后追踪 (模式1用)
        self._pid_chassis_dist = PID(kp=1.5, ki=0.0, kd=0.1, out_min=-1.0, out_max=1.0)
        # 脖子仰俯追踪 (模式2用)
        self._pid_neck_pitch = PID(kp=0.8, ki=0.0, kd=0.05, out_min=-1.0, out_max=1.0)

        # 内部仰俯状态 (-1.0: 最下, 1.0: 最上)
        self._current_neck_pitch = 0.0 

        # 状态机：抬头寻找人脸的持续时间
        self._search_face_tilt_timer = 0.0

        # ── 订阅与发布 ──
        if HAS_HOBOT_MSGS:
            self.create_subscription(PerceptionTargets, '/hobot_mono2d_body_detection', self._on_detection, 10)
        else:
            self.create_subscription(String, '/hobot_mono2d_body_detection', lambda x: None, 10)

        self.create_subscription(String, '/action_cmd', self._on_action_cmd, 10)

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
        self._last_detection_message = now
        dt = now - self._last_time
        self._last_time = now

        body_boxes = []
        face_boxes = []

        for target in msg.targets:
            for roi in target.rois:
                rect = roi.rect
                cx = rect.x_offset + rect.width / 2.0
                cy = rect.y_offset + rect.height / 2.0
                area_ratio = (rect.width * rect.height) / (self.IMG_WIDTH * self.IMG_HEIGHT)
                if roi.type in ["body", "person"]:
                    body_boxes.append((cx, cy, area_ratio))
                elif roi.type in ["face", "head"]:
                    face_boxes.append((cx, cy, area_ratio))

        if body_boxes or face_boxes:
            self._last_nonempty_detection = now

        if self.mode == self.MODE_BODY_FOLLOW:
            self._handle_body_follow(body_boxes, dt)
        elif self.mode == self.MODE_FACE_FOLLOW:
            self._handle_face_follow(face_boxes, body_boxes, dt)

    def _handle_body_follow(self, body_boxes, dt):
        """模式 1: 纯底盘跟随 (前进后退+左右转)，摄像头仰俯锁定平视"""
        if not body_boxes:
            return  # 丢失交由 _control_tick 处理原地打转

        self._mark_target_seen()
        best = self._largest_box(body_boxes)
        cx, cy, area_ratio = best

        # 因为图像被底层 flip_horizontal 翻转了，所以此处 X 误差必须取反，才能保证底盘转向正确的物理方向
        x_error = -(cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
        dist_error = self.BODY_TARGET_RATIO - area_ratio

        # 2. 误差死区，防止原地震荡抖动
        if abs(x_error) < 0.05:
            x_error = 0.0
        if abs(dist_error) < 0.05:
            dist_error = 0.0

        # 3. PID 计算底盘动力
        yaw_out = self._pid_chassis_yaw.update(x_error, dt)
        dist_out = self._pid_chassis_dist.update(dist_error, dt)

        # yaw_out > 0 表示人在右侧，需要右转
        left_throttle = dist_out + yaw_out
        right_throttle = dist_out - yaw_out

        # 限制在 -1.0 ~ 1.0 并转换到 0~100 的指令
        left_speed = max(min(left_throttle, 1.0), -1.0)
        right_speed = max(min(right_throttle, 1.0), -1.0)
        self._publish_motor_diff(left_speed, right_speed)

        # 3. 仿生虚假扭头 (x_error直接映射) + 强制平视
        self._publish_head_and_neck(x_error, pitch_val=0.0)


    def _handle_face_follow(self, face_boxes, body_boxes, dt):
        """模式 2: 禅定注视 (底盘静止，双舵机动态俯仰)"""
        target = None
        now = time.monotonic()
        
        # 1. 寻找目标
        if face_boxes:
            # Multiple people may be visible. Always look at the largest face,
            # which is normally the person closest to Wali.
            target = self._largest_box(face_boxes)
            self._search_face_tilt_timer = 0.0 # 清除抬头搜寻倒计时
        elif body_boxes:
            # 只看见身体没看见脸，尝试抬头搜寻
            if self._search_face_tilt_timer == 0.0:
                self._search_face_tilt_timer = now
                
            if now - self._search_face_tilt_timer < 2.0:
                # 前两秒内强制让脖子上扬，试图把脸纳入画面
                self._current_neck_pitch += 0.5 * dt
                self._current_neck_pitch = max(min(self._current_neck_pitch, 1.0), -1.0)
                # 因为没找到目标，所以水平偏差假定为身体的偏差来做仿生扭头
                best_body = self._largest_box(body_boxes)
                # 因为图像被底层 flip_horizontal 翻转了，这里取反
                x_error = -(best_body[0] - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
                self._publish_head_and_neck(x_error, self._current_neck_pitch)
                self._mark_target_seen()
                self._stop_motor() # 底盘死死刹住
                return
            else:
                # 抬头找了2秒还是没脸，妥协降级，直接把这个肚子当目标
                target = self._largest_box(body_boxes)

        if not target:
            return # 彻底丢失，交由 _control_tick 处理

        # 2. 锁定目标，进行调节
        self._mark_target_seen()
        cx, cy, area_ratio = target

        # 只要画面内有目标，底盘死死刹住
        self._stop_motor()

        # 垂直误差 (负=偏上需抬头, 正=偏下需低头)
        y_error = (cy - self.IMG_HEIGHT / 2.0) / (self.IMG_HEIGHT / 2.0)
        # 水平误差 (纯粹为了生动仿生头扭动) - 取反以补偿底层的左右镜像
        x_error = -(cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
        
        # 死区控制：如果误差很小，当作 0 处理，防止舵机疯狂抽搐
        if abs(x_error) < 0.05:
            x_error = 0.0
        if abs(y_error) < 0.05:
            y_error = 0.0

        # PID 计算脖子动力
        pitch_out = self._pid_neck_pitch.update(y_error, dt)

        # 累加到绝对俯仰角 (-1.0 最下 到 1.0 最上) 
        # 注意: y_error负数代表目标在上方，我们需要仰角变大(+)。所以减去y_error。
        self._current_neck_pitch += -pitch_out
        self._current_neck_pitch = max(min(self._current_neck_pitch, 1.0), -1.0)



        # 下发至 sequence_ros_node
        self._publish_head_and_neck(x_error, self._current_neck_pitch)


    def _control_tick(self):
        """Search briefly, then fail safe and eventually release the camera."""
        if self.mode == self.MODE_IDLE or self._joy_override:
            return

        self._renew_camera_lease()

        lost_seconds = time.monotonic() - self._last_target_seen
        if lost_seconds >= self.TRACKING_SHUTDOWN_DELAY_SEC:
            self.get_logger().warning(
                "目标丢失超过60秒，退出视觉跟随并关闭跟踪摄像头"
            )
            self._set_tracking_mode(self.MODE_IDLE)
            return

        self._warn_if_detection_is_missing(lost_seconds)

        # "look_at_me" is a stationary gaze mode. A detector warm-up or an
        # empty result must never make the chassis rotate; keep sending the
        # stop heartbeat so the motor watchdog also remains authoritative.
        if self.mode == self.MODE_FACE_FOLLOW:
            self._stop_motor()
            if lost_seconds >= self.SEARCH_STOP_DELAY_SEC and not self._search_halted:
                self._current_neck_pitch = 0.0
                self._publish_head_and_neck(x_error=0.0, pitch_val=0.0)
                self._search_halted = True
            return

        if lost_seconds >= self.SEARCH_STOP_DELAY_SEC:
            if not self._search_halted:
                self._stop_motor()
                self._current_neck_pitch = 0.0
                self._publish_head_and_neck(x_error=0.0, pitch_val=0.0)
                self._search_active = False
                self._search_halted = True
                self.get_logger().warning("目标丢失超过5秒，停止旋转搜索")
            return

        if lost_seconds >= self.SEARCH_START_DELAY_SEC:
            # Heartbeat the search command throughout the 1s..5s loss window.
            self._publish_motor(1, 2, self.SEARCH_ROTATE_SPEED)
            if not self._search_active:
                self._current_neck_pitch = 0.0
                self._publish_head_and_neck(x_error=0.0, pitch_val=0.0)
                self._search_active = True

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

    @staticmethod
    def _largest_box(boxes):
        return max(boxes, key=lambda box: box[2], default=None)

    def _mark_target_seen(self):
        self._last_target_seen = time.monotonic()
        self._search_active = False
        self._search_halted = False

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

        # 2. 脖子仰俯双舵机补偿
        # neck_top: 减小为抬头 (5000 -> 4000)
        # neck_bottom: 增大为抬头 (4000 -> 7000)，减小为低头 (4000 -> 3500)
        targets['neck_top'] = int(5000 - pitch_val * 1000)
        
        if pitch_val > 0:
            targets['neck_bottom'] = int(4000 + pitch_val * 3000)
        else:
            targets['neck_bottom'] = int(4000 + pitch_val * 500) # pitch_val为负，结果是减

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
            self._current_neck_pitch = 0.0
            now = time.monotonic()
            self._mode_started_at = now
            self._last_detection_message = 0.0
            self._last_nonempty_detection = 0.0
            self._last_detection_warning = 0.0
            self._last_time = now
            self._last_target_seen = now
            self._search_active = False
            self._search_halted = False
            self._pid_chassis_yaw.reset()
            self._pid_chassis_dist.reset()
            self._pid_neck_pitch.reset()
            # The camera manager is the sole V4L2 owner.  Acquire it before
            # starting consumers so the detector can wait for /image instead
            # of racing a second hobot_usb_cam instance.
            self._publish_camera_lease("acquire")
            self._last_camera_renew = now
            self.get_logger().info(f"Entered tracking mode: {mode} (requested: {mode_key})")
            return True
        elif mode == self.MODE_IDLE:
            self.mode = self.MODE_IDLE
            self._search_active = False
            self._search_halted = False
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

        if name == "set_tracking_mode":
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
