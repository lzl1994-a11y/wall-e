#!/usr/bin/env python3
"""Single-owner ROS bridge for direct Ubuntu I2C control of PCA9685."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.servo_control import ServoControl


class I2CHardwareNode(Node):
    def __init__(self):
        super().__init__("i2c_hardware_node")
        self.driver = ServoControl()
        self.create_subscription(String, "/servo_cmd", self._on_servo_cmd, 10)
        self.create_subscription(String, "/motor_cmd", self._on_motor_cmd, 10)
        self.get_logger().info(
            "Ubuntu I2C hardware online: "
            f"/dev/i2c-{self.driver.bus_number}, address=0x{self.driver.address:02x}, "
            f"frequency={self.driver.frequency}Hz"
        )

    def _on_servo_cmd(self, msg):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"/servo_cmd JSON 解析失败: {msg.data[:80]}")
            return

        name = str(command.get("name", ""))
        try:
            if "pwm" in command:
                accepted = self.driver.set_pwm(name, command["pwm"])
            elif "angle" in command:
                accepted = self.driver.set_angle(name, command["angle"])
            else:
                accepted = False
        except (TypeError, ValueError, OSError) as exc:
            self.get_logger().error(f"舵机 I2C 写入失败: {exc}")
            return
        if not accepted:
            self.get_logger().warn(f"未知或无效舵机指令: {command}")

    def _on_motor_cmd(self, msg):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"/motor_cmd JSON 解析失败: {msg.data[:80]}")
            return

        try:
            for side in ("left", "right"):
                motor = command.get(side, {})
                self.driver.set_motor(
                    side,
                    motor.get("action", 0),
                    motor.get("throttle", 0),
                )
        except (TypeError, ValueError, OSError) as exc:
            self.get_logger().error(f"电机 I2C 写入失败: {exc}")

    def destroy_node(self):
        if hasattr(self, "driver"):
            self.driver.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = I2CHardwareNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
