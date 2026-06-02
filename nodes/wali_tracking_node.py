#!/usr/bin/env python3
# nodes/wali_tracking_node.py
# 瓦力视觉跟踪中枢节点
#
# 订阅:
#   /hobot_mono2d_body_detection  (RDK X3 BPU 官方感知节点输出，ai_msgs/PerceptionTargets)
#   /action_cmd                    (LLM 大脑节点下发，std_msgs/String，JSON)
#   /doa_angle                     (DOA 声源定位角度，std_msgs/Int32)
#
# 发布:
#   /servo_cmd  (std_msgs/String, JSON)  -> servo_ros_node
#   /motor_cmd  (std_msgs/String, JSON)  -> motor_ros_node
#
# 模式状态机: IDLE -> BODY_FOLLOW / FACE_FOLLOW -> IDLE
# 入口: LLM 下发 "set_tracking_mode" action 或 "set_vision_gate"
# DOA 触发: 进入跟随模式时先根据声源角度做一次定向旋转

import json
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

# ---------------------------------------------------------------------------
# 尝试导入 RDK X3 的 ai_msgs，如果不在 RDK 上运行则用占位
# ---------------------------------------------------------------------------
try:
    from ai_msgs.msg import PerceptionTargets
    HAS_HOBOT_MSGS = True
except ImportError:
    HAS_HOBOT_MSGS = False

# ---------------------------------------------------------------------------
# 简易 PID 控制器
# ---------------------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False

    def update(self, error, dt):
        if not self._has_prev:
            self._has_prev = True
            self._prev_error = error
            return self.kp * error

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return max(self.out_min, min(self.out_max, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._has_prev = False


class WaliTrackingNode(Node):
    """瓦力视觉跟踪节点：感知 -> 决策 -> 舵机/电机指令"""

    # ── 跟踪模式常量 ──
    MODE_IDLE        = "idle"
    MODE_BODY_FOLLOW = "body_follow"
    MODE_FACE_FOLLOW = "face_follow"

    # ── 图像尺寸（与 mono2d_body_detection 配置一致）──
    IMG_WIDTH  = 960
    IMG_HEIGHT = 544

    # ── 控制参数（可在 config 中覆盖）──
    # 人体跟随
    BODY_YAW_KP       = 0.06    # 底盘旋转 PID 比例
    BODY_DIST_KP      = 0.04    # 前后速度比例
    BODY_TARGET_RATIO = 0.15    # 目标人体框面积占比（框面积/图像面积）
    BODY_ROTATE_DEAD  = 0.08    # 旋转死区（误差绝对值小于此值不转）
    BODY_MOVE_DEAD    = 0.03    # 前进后退死区
    # 人脸跟随
    FACE_YAW_KP       = 0.05    # 底盘补偿旋转比例
    FACE_PITCH_KP     = 0.7     # 脖子俯仰比例
    FACE_EDGE_MARGIN  = 0.15    # 人脸靠近画面边缘阈值（归一化）
    FACE_LOST_FRAMES  = 30      # 丢脸容忍帧数（~1秒@30fps）
    # 搜索
    SEARCH_ROTATE_SPEED = 25    # 搜索时旋转油门

    def __init__(self):
        super().__init__('wali_tracking_node')
        self.get_logger().info("Wali Tracking Node initializing...")

        # ── 状态 ──
        self.mode = self.MODE_IDLE
        self._doa_pending = False      # True: 等待 DOA 角度，先做一次定向
        self._doa_angle = None         # 最近收到的 DOA 角度
        self._face_lost_count = 0      # 人脸连续丢失帧计数
        self._last_time = time.time()

        # ── PID 控制器 ──
        self._pid_body_yaw = PID(
            kp=self.BODY_YAW_KP, ki=0.0, kd=0.02, out_min=-1.0, out_max=1.0
        )
        self._pid_body_dist = PID(
            kp=self.BODY_DIST_KP, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0
        )
        self._pid_face_yaw = PID(
            kp=self.FACE_YAW_KP, ki=0.0, kd=0.01, out_min=-1.0, out_max=1.0
        )
        self._pid_face_pitch = PID(
            kp=self.FACE_PITCH_KP, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0
        )

        # ── 订阅 ──
        if HAS_HOBOT_MSGS:
            self._det_sub = self.create_subscription(
                PerceptionTargets,
                '/hobot_mono2d_body_detection',
                self._on_detection,
                10
            )
            self.get_logger().info("Subscribed to /hobot_mono2d_body_detection (PerceptionTargets)")
        else:
            self.get_logger().warn(
                "ai_msgs not available. Subscribe /hobot_mono2d_body_detection as String fallback."
            )
            self._det_sub = self.create_subscription(
                String,
                '/hobot_mono2d_body_detection',
                self._on_detection_string,
                10
            )

        self._action_sub = self.create_subscription(
            String, '/action_cmd', self._on_action_cmd, 10
        )
        self._doa_sub = self.create_subscription(
            Int32, '/doa_angle', self._on_doa_angle, 10
        )

        # ── 发布 ──
        self._servo_pub = self.create_publisher(String, '/servo_cmd', 10)
        self._motor_pub = self.create_publisher(String, '/motor_cmd', 10)

        # ── 定时器：丢失目标时的搜索行为（10Hz）──
        self._timer = self.create_timer(0.1, self._control_tick)

        self.get_logger().info("Wali Tracking Node ready. Mode: IDLE")

    # ===================================================================
    # 回调: 检测结果
    # ===================================================================

    def _on_detection(self, msg):
        """解析 ai_msgs/PerceptionTargets 消息"""
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        if self.mode == self.MODE_IDLE:
            return

        # 解析所有检测到的目标
        body_boxes = []  # (center_x, center_y, area_ratio, track_id)
        face_boxes = []  # (center_x, center_y, area_ratio, track_id)

        for target in msg.targets:
            track_id = target.track_id
            for roi in target.rois:
                rect = roi.rect
                cx = rect.x_offset + rect.width / 2.0
                cy = rect.y_offset + rect.height / 2.0
                area_ratio = (rect.width * rect.height) / (self.IMG_WIDTH * self.IMG_HEIGHT)
                if roi.type == "body":
                    body_boxes.append((cx, cy, area_ratio, track_id))
                elif roi.type == "face":
                    face_boxes.append((cx, cy, area_ratio, track_id))

        if self.mode == self.MODE_BODY_FOLLOW:
            self._handle_body_follow(body_boxes, dt)

        elif self.mode == self.MODE_FACE_FOLLOW:
            self._handle_face_follow(face_boxes, body_boxes, dt)

    def _on_detection_string(self, msg):
        """备用：当 ai_msgs 不可用时打印原始消息供调试"""
        if self.mode != self.MODE_IDLE:
            self.get_logger().debug(f"Raw detection: {msg.data[:200]}")

    # ===================================================================
    # 回调: LLM 动作指令
    # ===================================================================

    def _on_action_cmd(self, msg):
        """解析 LLM 下发的 action_cmd JSON"""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        name = payload.get("name", "")
        args_str = payload.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}

        # LLM 下发 set_tracking_mode: 切换跟随模式
        if name == "set_tracking_mode":
            mode = args.get("mode", "")
            if mode in (self.MODE_BODY_FOLLOW, self.MODE_FACE_FOLLOW):
                self._enter_tracking_mode(mode)
            elif mode == self.MODE_IDLE:
                self._exit_tracking_mode()

        # LLM 下发 set_vision_gate: 开关视觉
        elif name == "set_vision_gate":
            enabled = args.get("enabled", False)
            if enabled:
                # 默认进入人体跟随
                self._enter_tracking_mode(self.MODE_BODY_FOLLOW)
            else:
                self._exit_tracking_mode()

    # ===================================================================
    # 回调: DOA 声源角度
    # ===================================================================

    def _on_doa_angle(self, msg):
        self._doa_angle = msg.data
        self.get_logger().info(f"DOA angle received: {self._doa_angle}°")

    # ===================================================================
    # 模式切换
    # ===================================================================

    def _enter_tracking_mode(self, mode):
        """进入跟踪模式，先触发 DOA 定向"""
        self.mode = mode
        self._face_lost_count = 0
        self._pid_body_yaw.reset()
        self._pid_body_dist.reset()
        self._pid_face_yaw.reset()
        self._pid_face_pitch.reset()
        self._doa_pending = True

        # 脖子回中立
        self._publish_servo("neck_top", 90)
        self._publish_servo("neck_bottom", 90)

        mode_label = "人体跟随" if mode == self.MODE_BODY_FOLLOW else "人脸跟随"
        self.get_logger().info(f"Entering tracking mode: {mode_label}")

        # 如果有 DOA 角度，立刻做一次定向旋转
        if self._doa_angle is not None:
            self._doa_orient(self._doa_angle)
        else:
            self._doa_pending = False

    def _exit_tracking_mode(self):
        """退出跟踪，停止所有运动"""
        self.mode = self.MODE_IDLE
        self._doa_pending = False
        self._stop_all()
        self.get_logger().info("Exited tracking mode. All motors stopped.")

    def _doa_orient(self, angle_deg):
        """
        根据 DOA 声源角度做一次性定向：旋转底盘 + 头部拟人化转向。
        angle_deg: 声音来源角度（0=正前方，负=左侧，正=右侧）
        """
        # 底盘：差速原地旋转（左正右负）
        rotate_time = abs(angle_deg) / 180.0 * 1.5  # 1.5秒转180度
        rotate_dir = 1 if angle_deg > 0 else -1     # 正=右转, 负=左转
        self._publish_motor(rotate_dir, rotate_dir, self.SEARCH_ROTATE_SPEED)
        time.sleep(min(rotate_time, 2.0))
        self._publish_motor(0, 0, 0)

        # 头部拟人化：随声音方向转动 head_yaw
        target_yaw = 90 + int(angle_deg * 0.5)  # 映射到舵机角度
        target_yaw = max(30, min(150, target_yaw))
        self._publish_servo("head_yaw", target_yaw)

        self._doa_pending = False
        self.get_logger().info(f"DOA orient completed: angle={angle_deg}°")

    # ===================================================================
    # 人体跟随逻辑
    # ===================================================================

    def _handle_body_follow(self, body_boxes, dt):
        """
        人体跟随策略:
        - 选最大的 body 框作为目标
        - 水平误差 -> 底盘差速旋转，保持人在画面中央
        - 面积误差 -> 前进/后退，保持期望距离
        - 脖子保持中立
        """
        if not body_boxes:
            # 丢失目标 -> 底盘原地慢旋搜索
            self._publish_motor(1, 2, self.SEARCH_ROTATE_SPEED)  # 左前右后 = 原地右转
            return

        # 选面积最大的 body 框
        best = max(body_boxes, key=lambda b: b[2])
        cx, cy, area_ratio, track_id = best

        # 水平误差: 归一化 [-1, 1]，正=目标偏右
        x_error = (cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)

        # 距离误差: 正=太远需前进，负=太近需后退
        dist_error = self.BODY_TARGET_RATIO - area_ratio

        # PID 计算
        yaw_out = self._pid_body_yaw.update(x_error, dt)
        dist_out = self._pid_body_dist.update(dist_error, dt)

        # 死区过滤
        if abs(x_error) < self.BODY_ROTATE_DEAD:
            yaw_out = 0.0
        if abs(dist_error) < self.BODY_MOVE_DEAD:
            dist_out = 0.0

        # 转换为底盘指令（差速驱动）
        base_throttle = 30   # 基础油门
        rotate_throttle = int(abs(yaw_out) * base_throttle)
        move_throttle   = int(abs(dist_out) * 40)

        if abs(yaw_out) > 0.01:
            # 旋转优先：原地差速旋转
            if yaw_out > 0:
                self._publish_motor(1, 2, rotate_throttle)  # 右转
            else:
                self._publish_motor(2, 1, rotate_throttle)  # 左转
        elif abs(dist_out) > 0.01:
            # 直行
            if dist_out > 0:
                self._publish_motor(1, 1, move_throttle)    # 前进
            else:
                self._publish_motor(2, 2, move_throttle)    # 后退
        else:
            self._publish_motor(0, 0, 0)  # 停止

    # ===================================================================
    # 人脸跟随逻辑
    # ===================================================================

    def _handle_face_follow(self, face_boxes, body_boxes, dt):
        """
        人脸跟随策略:
        - 人脸在画面中部 -> 底盘停止，脖子 pitch 跟踪人脸垂直位置
        - 人脸靠近画面边缘 -> 底盘慢速差速补偿
        - 人脸丢失 -> 降级用 body 框补偿；全丢则慢旋搜索
        """
        if face_boxes:
            self._face_lost_count = 0
            # 选最大的人脸框
            best = max(face_boxes, key=lambda f: f[2])
            cx, cy, area_ratio, track_id = best

            # 水平归一化误差
            x_error = (cx - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
            # 垂直归一化误差: 负=人脸偏上，正=人脸偏下
            y_error = (cy - self.IMG_HEIGHT / 2.0) / (self.IMG_HEIGHT / 2.0)

            abs_x = abs(x_error)

            if abs_x < self.FACE_EDGE_MARGIN:
                # 人脸在画面中部 -> 底盘停止，脖子做垂直跟踪
                self._publish_motor(0, 0, 0)

                pitch_out = self._pid_face_pitch.update(y_error, dt)
                # 脖子角度: 中立90，上仰>90，下俯<90
                neck_angle = 90 - int(pitch_out * 30)  # ±30度范围
                neck_angle = max(60, min(120, neck_angle))
                self._publish_servo("neck_top", neck_angle)
                self._publish_servo("neck_bottom", neck_angle)

            else:
                # 人脸偏出中部 -> 底盘慢速差速补偿
                yaw_out = self._pid_face_yaw.update(x_error, dt)
                comp_throttle = int(abs(yaw_out) * 20) + 10
                if yaw_out > 0:
                    self._publish_motor(1, 2, comp_throttle)  # 右转补偿
                else:
                    self._publish_motor(2, 1, comp_throttle)  # 左转补偿

                # 脖子同时尝试跟踪垂直
                pitch_out = self._pid_face_pitch.update(y_error, dt)
                neck_angle = 90 - int(pitch_out * 30)
                neck_angle = max(60, min(120, neck_angle))
                self._publish_servo("neck_top", neck_angle)
                self._publish_servo("neck_bottom", neck_angle)

        else:
            # 人脸丢失
            self._face_lost_count += 1
            if self._face_lost_count > self.FACE_LOST_FRAMES:
                # 超时，脖子回中立，底盘慢旋搜索
                self._publish_servo("neck_top", 90)
                self._publish_servo("neck_bottom", 90)
                if body_boxes:
                    # 用 body 框辅助找到大致方向
                    best_body = max(body_boxes, key=lambda b: b[2])
                    bx_err = (best_body[0] - self.IMG_WIDTH / 2.0) / (self.IMG_WIDTH / 2.0)
                    if bx_err > 0.1:
                        self._publish_motor(1, 2, self.SEARCH_ROTATE_SPEED)
                    elif bx_err < -0.1:
                        self._publish_motor(2, 1, self.SEARCH_ROTATE_SPEED)
                    else:
                        self._publish_motor(0, 0, 0)
                else:
                    self._publish_motor(1, 2, self.SEARCH_ROTATE_SPEED)

    # ===================================================================
    # 定时器: 搜索行为 & 安全超时
    # ===================================================================

    def _control_tick(self):
        """10Hz 定时回调，处理 DOA 等待和 IDLE 状态下的残留"""
        if self.mode == self.MODE_IDLE:
            return  # IDLE 下不主动发任何指令

        if self._doa_pending and self._doa_angle is not None:
            # 进入模式后 DOA 数据到达，执行一次定向
            self._doa_orient(self._doa_angle)

    # ===================================================================
    # 发布辅助
    # ===================================================================

    def _publish_servo(self, name, angle):
        """发布舵机指令到 /servo_cmd"""
        msg = String()
        msg.data = json.dumps({"name": name, "angle": int(angle)})
        self._servo_pub.publish(msg)

    def _publish_motor(self, left_action, right_action, throttle):
        """
        发布电机指令到 /motor_cmd
        left_action, right_action: 0=停, 1=正转, 2=反转
        throttle: 0-100
        """
        msg = String()
        msg.data = json.dumps({
            "left":  {"action": left_action,  "throttle": throttle},
            "right": {"action": right_action, "throttle": throttle}
        })
        self._motor_pub.publish(msg)

    def _stop_all(self):
        """停止所有舵机和电机"""
        self._publish_servo("neck_top", 90)
        self._publish_servo("neck_bottom", 90)
        self._publish_servo("head_yaw", 90)
        self._publish_motor(0, 0, 0)

    def destroy_node(self):
        self._stop_all()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WaliTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()