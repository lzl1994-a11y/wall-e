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
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

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

    def __init__(self):
        super().__init__('wali_tracking_node')

        self.mode = self.MODE_IDLE
        self._last_time = time.time()
        self._joy_override = False # 若未来恢复 joy_override 机制

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

        self._action_pub = self.create_publisher(String, '/action_cmd', 10)
        self._motor_pub = self.create_publisher(String, '/motor_cmd', 10)

        # 丢失目标的搜索定时器
        self._timer = self.create_timer(0.1, self._control_tick)
        self._last_target_seen = time.time()

        self.get_logger().info("视觉跟踪节点上线 (双舵机俯仰 + 底盘左右版本)")

    # ===================================================================
    # 核心检测处理
    # ===================================================================
    def _on_detection(self, msg):
        if self.mode == self.MODE_IDLE or self._joy_override:
            return

        now = time.time()
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

        if self.mode == self.MODE_BODY_FOLLOW:
            self._handle_body_follow(body_boxes, dt)
        elif self.mode == self.MODE_FACE_FOLLOW:
            self._handle_face_follow(face_boxes, body_boxes, dt)

    def _handle_body_follow(self, body_boxes, dt):
        """模式 1: 纯底盘跟随 (前进后退+左右转)，摄像头仰俯锁定平视"""
        if not body_boxes:
            return  # 丢失交由 _control_tick 处理原地打转

        self._last_target_seen = time.time()
        best = max(body_boxes, key=lambda b: b[2])
        cx, cy, area_ratio = best

        x_error = (cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
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
        now = time.time()
        
        # 1. 寻找目标
        if face_boxes:
            target = max(face_boxes, key=lambda b: b[2])
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
                best_body = max(body_boxes, key=lambda b: b[2])
                x_error = (best_body[0] - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
                self._publish_head_and_neck(x_error, self._current_neck_pitch)
                self._last_target_seen = time.time()
                self._stop_motor() # 底盘死死刹住
                return
            else:
                # 抬头找了2秒还是没脸，妥协降级，直接把这个肚子当目标
                target = max(body_boxes, key=lambda b: b[2])

        if not target:
            return # 彻底丢失，交由 _control_tick 处理

        # 2. 锁定目标，进行调节
        self._last_target_seen = time.time()
        cx, cy, area_ratio = target

        # 只要画面内有目标，底盘死死刹住
        self._stop_motor()

        # 垂直误差 (负=偏上需抬头, 正=偏下需低头)
        y_error = (cy - self.IMG_HEIGHT / 2.0) / (self.IMG_HEIGHT / 2.0)
        # 水平误差 (纯粹为了生动仿生头扭动)
        x_error = (cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
        
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
        """低频检查：丢失目标的打转搜寻逻辑"""
        if self.mode == self.MODE_IDLE or self._joy_override:
            return

        if time.time() - self._last_target_seen > 1.0:
            # 超过1秒没看到目标，底盘原地向右缓慢打转
            # 左前，右后 = 右转
            self._publish_motor(1, 2, self.SEARCH_ROTATE_SPEED)
            # 脑袋和脖子复位
            self._current_neck_pitch = 0.0
            self._publish_head_and_neck(x_error=0.0, pitch_val=0.0)

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

        # 发送到 /action_cmd 的 manual_servo 接口
        payload = {
            "name": "manual_servo",
            "arguments": {
                "targets": targets,
                "step_size": 40.0
            }
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._action_pub.publish(msg)

    def _set_tracking_mode(self, requested_mode):
        mode_key = str(requested_mode or "").strip()
        mode = self.MODE_ALIASES.get(mode_key, mode_key)

        if mode in (self.MODE_BODY_FOLLOW, self.MODE_FACE_FOLLOW):
            self.mode = mode
            self._current_neck_pitch = 0.0
            self._last_time = time.time()
            self._last_target_seen = time.time()
            self._pid_chassis_yaw.reset()
            self._pid_chassis_dist.reset()
            self._pid_neck_pitch.reset()
            self.get_logger().info(f"Entered tracking mode: {mode} (requested: {mode_key})")
        elif mode == self.MODE_IDLE:
            self.mode = self.MODE_IDLE
            self._stop_motor()
            self._publish_head_and_neck(0.0, 0.0) # 回中
            self.get_logger().info("Tracking mode: IDLE")
        else:
            self.get_logger().warn(f"Unknown tracking mode: {requested_mode}")

    # ===================================================================
    # 模式切换监听
    # ===================================================================
    def _on_action_cmd(self, msg):
        try:
            payload = json.dumps(msg.data) if not isinstance(msg.data, dict) else msg.data
            payload = json.loads(msg.data)
        except: return

        name = payload.get("name", "")
        args_str = payload.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except: args = {}

        if name == "set_tracking_mode":
            self._set_tracking_mode(args.get("mode", ""))
        elif name == "set_vision_gate":
            enabled = args.get("enabled", False)
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
            self._set_tracking_mode(self.MODE_BODY_FOLLOW if enabled else self.MODE_IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = WaliTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_motor()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
