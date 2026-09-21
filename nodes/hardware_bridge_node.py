#!/usr/bin/env python3
"""硬件桥接节点：合并舵机与电机订阅，计算 PCA9685 原始值发布到 /pca9685_raw。

本节点不直接持有串口；串口由 serial_ros_node 独占。它只把算好的 15 通道值
通过 ROS Topic 交给 serial_ros_node 透传 ESP32-S3。

协议格式（/pca9685_raw 内 payload）：
  ch0,ch1,...,ch8, ch9,ch10,ch11, ch12,ch13,ch14
  0-8:  PCA9685 OFF 寄存器值 (1638~8192 对应 0~180°)
  9-14: PCA9685 duty_cycle 值 (0~65535)

电机通道布局（与 ServoControl 一致）：
  左电机: ch9=IN1, ch10=IN2, ch11=PWM
  右电机: ch12=IN1, ch13=IN2, ch14=PWM
"""

import json
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml

from services.motor_control import apply_direction_inversion, motor_inversion_flags
from services.motion_arbiter import normalize_motor_command
from services.motor_watchdog import MotorWatchdog
from services.pca9685_output_state import Pca9685OutputState

class HardwareBridgeNode(Node):
    _PUBLISH_INTERVAL_SECONDS = 0.02

    def __init__(self):
        super().__init__('hardware_bridge_node')

        self._name_to_ch = {}
        self._motor_inverted = {"left": False, "right": False}
        servo_inits = {}
        try:
            yaml_path = os.path.join(os.path.dirname(__file__), '../core/config.yaml')
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                for servo in config_data.get('servos', []):
                    idx = servo.get('id')
                    s_name = servo.get('name')
                    if idx is not None and s_name is not None:
                        self._name_to_ch[s_name] = idx
                        init_val = servo.get('init')
                        if init_val is not None and 0 <= idx < 15:
                            servo_inits[idx] = int(init_val)
                self._motor_inverted = motor_inversion_flags(config_data.get('motors'))
        except Exception as e:
            self.get_logger().error(f'[Bridge] 读取 config.yaml 失败: {e}')

        self._state = Pca9685OutputState(servo_inits)
        self._motor_watchdog = MotorWatchdog()

        self.create_subscription(String, '/servo_cmd', self._on_servo_cmd, 10)
        self.create_subscription(String, '/motor_cmd', self._on_motor_cmd, 10)
        self._raw_pub = self.create_publisher(String, '/pca9685_raw', 10)
        self._publish_timer = self.create_timer(
            self._PUBLISH_INTERVAL_SECONDS, self._flush_state
        )

        self.get_logger().info(
            '硬件桥接节点上线，输出 -> /pca9685_raw '
            f'(电机反向: left={self._motor_inverted["left"]}, right={self._motor_inverted["right"]})'
        )

    # ------------------------------------------------------------------
    # Topic 发送
    # ------------------------------------------------------------------
    def _publish_state(self):
        msg = String()
        msg.data = self._state.encode()
        self._raw_pub.publish(msg)

    def _flush_state(self):
        """Publish at most one complete state packet per control frame."""
        if self._motor_watchdog.poll():
            self._state.stop_motors()
            self.get_logger().error("[Bridge] 电机指令超时，硬件后端已强制停车")
        if not self._state.dirty:
            return

        self._publish_state()
        self._state.mark_published()

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

        ch = self._name_to_ch.get(name)
        if ch is None:
            self.get_logger().warn(f'[Bridge] 未知舵机名称: {name}')
            return

        if pwm >= 0:
            self._state.set_pwm(ch, pwm)
        elif angle >= 0:
            self._state.set_angle(ch, angle)
        else:
            return

    def _on_motor_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f'[Bridge] motor JSON 解析失败: {msg.data[:80]}')
            return

        cmd = normalize_motor_command(cmd)
        if cmd is None:
            self.get_logger().warn(f'[Bridge] motor 指令无效: {msg.data[:80]}')
            return

        left = cmd.get('left', {})
        right = cmd.get('right', {})

        left_action = apply_direction_inversion(
            left.get('action', 0), self._motor_inverted['left']
        )
        right_action = apply_direction_inversion(
            right.get('action', 0), self._motor_inverted['right']
        )
        self._state.set_motor('left', left_action, left.get('throttle', 0))
        self._state.set_motor('right', right_action, right.get('throttle', 0))
        if self._motor_watchdog.refresh():
            self.get_logger().info('[Bridge] 电机心跳恢复')


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
