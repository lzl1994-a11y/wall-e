#!/usr/bin/env python3
"""硬件桥接节点：合并舵机与电机订阅，计算 PCA9685 原始值发布到 /pca9685_raw。

替换 servo_ros_node + motor_ros_node。不再直接持有串口——串口由 serial_ros_node 独占，
本节点只把算好的 15 通道值通过 ROS Topic 交给 serial_ros_node 透传 ESP32-S3。

协议格式（/pca9685_raw 内 payload）：
  ch0,ch1,...,ch8, ch9,ch10,ch11, ch12,ch13,ch14
  0-8:  PCA9685 OFF 寄存器值 (1638~8192 对应 0~180°)
  9-14: PCA9685 duty_cycle 值 (0~65535)

电机通道布局（与 ServoControl 一致）：
  左电机: ch9=IN1, ch10=IN2, ch11=PWM
  右电机: ch12=IN1, ch13=IN2, ch14=PWM
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HardwareBridgeNode(Node):
    def __init__(self):
        super().__init__('hardware_bridge_node')

        # 角度→占空比换算常量 (50Hz / 16-bit)
        self._DUTY_MIN = 1638
        self._DUTY_MAX = 8192

        # 电机 ALL_HIGH / ALL_LOW
        self._MOTOR_HIGH = 65535
        self._MOTOR_LOW = 0

        # 15 通道当前状态 (PCA9685 原始值)
        # 舵机初始化为 90° (中性位)
        _90deg = int(self._DUTY_MIN + (self._DUTY_MAX - self._DUTY_MIN) * 90 / 180)
        self._state = [
            _90deg, _90deg, _90deg, _90deg,   # 0-3: 眉毛/眼睛
            _90deg, _90deg, _90deg, _90deg, _90deg,  # 4-8: 脖子/手臂
            0, 0, 0,    # 9-11: 左电机 (停止)
            0, 0, 0,    # 12-14: 右电机 (停止)
        ]

        self.create_subscription(String, '/servo_cmd', self._on_servo_cmd, 10)
        self.create_subscription(String, '/motor_cmd', self._on_motor_cmd, 10)
        self._raw_pub = self.create_publisher(String, '/pca9685_raw', 10)

        self.get_logger().info('硬件桥接节点上线，输出 -> /pca9685_raw')

    # ------------------------------------------------------------------
    # 角度换算 (与 ServoControl._angle_to_duty 完全一致)
    # ------------------------------------------------------------------
    def _angle_to_duty(self, angle: float) -> int:
        return int(self._DUTY_MIN + (self._DUTY_MAX - self._DUTY_MIN) * angle / 180)

    # ------------------------------------------------------------------
    # Topic 发送
    # ------------------------------------------------------------------
    def _publish_state(self):
        msg = String()
        msg.data = 'pca9685:' + ','.join(str(v) for v in self._state)
        self._raw_pub.publish(msg)

    # ------------------------------------------------------------------
    # 订阅回调
    # ------------------------------------------------------------------
    def _on_servo_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'[Bridge] servo JSON 解析失败: {msg.data[:80]}')
            return

        name = cmd.get('name', '')
        angle = cmd.get('angle', -1)
        pwm = cmd.get('pwm', -1)
        if not name:
            return

        ch = {
            "eyebrow_r": 0, "eyebrow_l": 1, "eye_r": 2, "eye_l": 3,
            "head_yaw": 4, "neck_top": 5, "neck_bottom": 6, "arm_r": 7, "arm_l": 8,
        }.get(name)
        if ch is None:
            self.get_logger().warn(f'[Bridge] 未知舵机名称: {name}')
            return

        if pwm >= 0:
            # 协议已统一为 16-bit 原始值，直接透传
            self._state[ch] = int(pwm)
        elif angle >= 0:
            self._state[ch] = self._angle_to_duty(angle)
            
        self._publish_state()

    def _on_motor_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'[Bridge] motor JSON 解析失败: {msg.data[:80]}')
            return

        left = cmd.get('left', {})
        right = cmd.get('right', {})

        self._apply_motor(9, left.get('action', 0), left.get('throttle', 0))
        self._apply_motor(12, right.get('action', 0), right.get('throttle', 0))
        self._publish_state()

    def _apply_motor(self, base_ch: int, action: int, throttle: int):
        """将一路电机的 action/throttle 写入 _state 对应 3 个通道。"""
        in1_ch, in2_ch, pwm_ch = base_ch, base_ch + 1, base_ch + 2

        if action == 1:          # 正转
            self._state[in1_ch] = self._MOTOR_HIGH
            self._state[in2_ch] = self._MOTOR_LOW
        elif action == 2:        # 反转
            self._state[in1_ch] = self._MOTOR_LOW
            self._state[in2_ch] = self._MOTOR_HIGH
        else:                    # 停止
            self._state[in1_ch] = self._MOTOR_LOW
            self._state[in2_ch] = self._MOTOR_LOW
            throttle = 0

        self._state[pwm_ch] = int(throttle / 100.0 * self._MOTOR_HIGH)


def main(args=None):
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
