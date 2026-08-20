#!/usr/bin/env python3
"""Single authority for motor priority, validation, and upstream timeout."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.motion_arbiter import (
    MOTOR_OUTPUT_TOPIC,
    OUTPUT_INTERVAL_SEC,
    SOURCE_TOPICS,
    MotionArbiter,
)


class MotionArbiterNode(Node):
    def __init__(self):
        super().__init__("motion_arbiter_node")
        self._arbiter = MotionArbiter()
        self._publisher = self.create_publisher(String, MOTOR_OUTPUT_TOPIC, 10)
        self._last_source = None
        self._subscriptions = [
            self.create_subscription(
                String,
                topic,
                lambda message, source=source: self._on_command(source, message),
                10,
            )
            for source, topic in SOURCE_TOPICS.items()
        ]
        self._timer = self.create_timer(OUTPUT_INTERVAL_SEC, self._publish_selected)
        self.get_logger().info(
            "运动仲裁器上线：joystick > tracking > autonomy，"
            "上游命令超时自动停车"
        )

    def _on_command(self, source, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning(f"拒绝无效 {source} 电机 JSON")
            return
        if not self._arbiter.update(source, payload):
            self.get_logger().warning(f"拒绝无效 {source} 电机指令")

    def _publish_selected(self):
        source, command = self._arbiter.select()
        self._publisher.publish(
            String(data=json.dumps(command, ensure_ascii=False, separators=(",", ":")))
        )
        if source != self._last_source:
            self.get_logger().info(f"电机控制权切换为: {source}")
            self._last_source = source

    def destroy_node(self):
        self._publisher.publish(
            String(data='{"left":{"action":0,"throttle":0},"right":{"action":0,"throttle":0}}')
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
