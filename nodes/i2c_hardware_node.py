#!/usr/bin/env python3
"""Single-owner ROS bridge for direct Ubuntu I2C control of PCA9685."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.servo_control import ServoControl
from services.motion_arbiter import normalize_motor_command
from services.motor_watchdog import (
    MOTOR_WATCHDOG_CHECK_INTERVAL_SEC,
    MotorWatchdog,
)


class I2CHardwareNode(Node):
    def __init__(self):
        super().__init__("i2c_hardware_node")
        self.driver = ServoControl()
        self._motor_watchdog = MotorWatchdog()
        self.create_subscription(String, "/servo_cmd", self._on_servo_cmd, 10)
        self.create_subscription(String, "/motor_cmd", self._on_motor_cmd, 10)
        self.create_timer(
            MOTOR_WATCHDOG_CHECK_INTERVAL_SEC,
            self._check_motor_watchdog,
        )
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

        command = normalize_motor_command(command)
        if command is None:
            self.get_logger().warn(f"/motor_cmd 指令无效: {msg.data[:80]}")
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
            return
        if self._motor_watchdog.refresh():
            self.get_logger().info("I2C 电机心跳恢复")

    def _check_motor_watchdog(self):
        if not self._motor_watchdog.poll():
            return
        try:
            self.driver.set_motor("left", 0, 0)
            self.driver.set_motor("right", 0, 0)
            self.get_logger().error("I2C 电机指令超时，硬件后端已强制停车")
        except (TypeError, ValueError, OSError) as exc:
            self.get_logger().error(f"I2C watchdog 停车失败: {exc}")

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
