#!/usr/bin/env python3
"""ROS adapter for the game lifecycle's state-only coordination boundary."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.game_mode import GameModeController, InvalidGameTransition


class GameModeNode(Node):
    """Own game state; hardware adapters subscribe instead of being called here."""

    def __init__(self):
        super().__init__("game_mode_node")
        self._controller = GameModeController()
        self._state_pub = self.create_publisher(String, "game_mode_state", 10)
        self.create_subscription(String, "game_mode_request", self._on_request, 10)
        self._publish_state()
        self.create_timer(1.0, self._publish_state)

    def _on_request(self, message):
        try:
            request = json.loads(message.data).get("request")
        except (TypeError, json.JSONDecodeError):
            return
        try:
            if request == "toggle" and self._controller.mode.value == "robot":
                self._controller.request_enter()
            elif request == "game_surface_ready":
                self._controller.game_surface_ready()
            elif request == "start_game":
                self._controller.start_game()
            elif request == "pause":
                self._controller.pause_for_fault()
            elif request == "resume":
                self._controller.resume_game()
            elif request == "toggle":
                self._controller.request_exit()
            elif request == "robot_surface_ready":
                self._controller.robot_surface_ready()
            else:
                return
        except InvalidGameTransition as exc:
            self.get_logger().warning(str(exc))
            return
        self._publish_state()

    def _publish_state(self):
        policy = self._controller.policy
        self._state_pub.publish(String(data=json.dumps({
            "mode": self._controller.mode.value,
            "robot_input": policy.robot_input,
            "recording": policy.recording,
            "game_input": policy.game_input,
        })))


def main(args=None):
    rclpy.init(args=args)
    node = GameModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
