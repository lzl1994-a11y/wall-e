#!/usr/bin/env python3
# nodes/motor_ros_node.py
# 电机 ROS 桥接节点
# 订阅 /motor_cmd (JSON: {"left":{"action":1,"throttle":50},"right":{...}})
# 调用 ServoControl.set_motor() 驱动 TB6612FNG

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from services.servo_control import ServoControl


class MotorRosNode(Node):
    """电机桥接节点：ROS topic -> ServoControl 底盘驱动"""

    def __init__(self):
        super().__init__('motor_ros_node')

        # 初始化底层舵机/电机驱动（PCA9685 统一管理）
        self.servo = ServoControl()
        self.get_logger().info("Motor ROS node started, listening on /motor_cmd")

        # 订阅电机指令
        self._sub = self.create_subscription(
            String, '/motor_cmd', self._on_motor_cmd, 10
        )

    def _on_motor_cmd(self, msg):
        """解析 JSON 电机指令并执行"""
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Invalid JSON on /motor_cmd: {msg.data}")
            return

        # 左电机
        left = cmd.get("left", {})
        self.servo.set_motor(
            "track_l",
            left.get("action", 0),
            left.get("throttle", 0)
        )
        # 右电机
        right = cmd.get("right", {})
        self.servo.set_motor(
            "track_r",
            right.get("action", 0),
            right.get("throttle", 0)
        )

        self.get_logger().debug(
            f"Motors: L(a={left.get('action',0)} t={left.get('throttle',0)}) "
            f"R(a={right.get('action',0)} t={right.get('throttle',0)})"
        )

    def destroy_node(self):
        self.servo.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()