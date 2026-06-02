#!/usr/bin/env python3
# nodes/servo_ros_node.py
# 舵机 ROS 桥接节点
# 订阅 /servo_cmd (JSON: {"name": "head_yaw", "angle": 110})，调用 ServoControl.set_angle()

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from services.servo_control import ServoControl


class ServoRosNode(Node):
    """舵机桥接节点：ROS topic -> ServoControl 硬件驱动"""

    def __init__(self):
        super().__init__('servo_ros_node')

        # 初始化底层舵机驱动（包含 PCA9685 I2C 初始化）
        self.servo = ServoControl()
        self.get_logger().info("Servo ROS node started, listening on /servo_cmd")

        # 订阅舵机指令
        self._sub = self.create_subscription(
            String, '/servo_cmd', self._on_servo_cmd, 10
        )

    def _on_servo_cmd(self, msg):
        """解析 JSON 舵机指令并执行"""
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Invalid JSON on /servo_cmd: {msg.data}")
            return

        name = cmd.get("name", "")
        angle = cmd.get("angle", 90)

        if name:
            self.servo.set_angle(name, angle)
            self.get_logger().debug(f"Servo {name} -> {angle}°")
        else:
            self.get_logger().warn(f"Servo command missing 'name': {cmd}")

    def destroy_node(self):
        self.servo.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()