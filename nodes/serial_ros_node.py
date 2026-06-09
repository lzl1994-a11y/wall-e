#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.serial_bridge import SerialBridge


class SerialNode(Node):
    def __init__(self):
        super().__init__('walle_serial_node')

        self.get_logger().info('Serial bridge node starting...')
        self.bridge = SerialBridge(device_name="WALL_E_TFT")

        if not self.bridge.ser:
            self.get_logger().error('Serial bridge connection failed; check hardware connection.')

        self.get_logger().info('Serial ROS node is online.')

        # Use only the atomic per-turn message for the screen to avoid duplicated
        # corrected_text/full_ai_text/action_cmd deliveries.
        self.sub_screen_dialog = self.create_subscription(
            String, 'screen_dialog', self.screen_dialog_callback, 10)

    def screen_dialog_callback(self, msg):
        """Send a complete turn to the lower screen in one callback."""
        try:
            dialog = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"screen_dialog JSON parse failed: {msg.data}")
            return

        turn_id = dialog.get("turn_id", "")
        corrected_text = (dialog.get("corrected_text") or "").strip()
        ai_text = (dialog.get("ai_text") or "").strip()
        actions = dialog.get("actions") or []

        if corrected_text:
            payload = f"you:{corrected_text}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'[{turn_id}] Sent user text -> {payload.strip()}')

        if ai_text:
            self.bridge.send_raw("openchat:1\n")
            self.bridge.send_raw("eyeaction:talk\n")
            payload = f"ai:{ai_text}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'[{turn_id}] Sent AI text -> {payload.strip()}')

        for action in actions:
            self._send_action_dict(action)

    def you_callback(self, msg):
        payload = f"you:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent user text -> {payload.strip()}')

    def ai_callback(self, msg):
        """Handle a full AI response from the legacy topic."""
        self.bridge.send_raw("openchat:1\n")
        self.bridge.send_raw("eyeaction:talk\n")
        payload = f"ai:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent AI text -> {payload.strip()}')

    def _send_action_dict(self, cmd_data):
        """Shared action translation for action_cmd and screen_dialog.actions."""
        tool_name = cmd_data.get("name")

        if tool_name == "express_emotion":
            args_obj = cmd_data.get("arguments", {})
            if isinstance(args_obj, str):
                args = json.loads(args_obj or "{}")
            else:
                args = args_obj
            emotion = args.get("emotion", "happy")
            payload = f"eyeaction:{emotion}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'Sent eye action -> {payload.strip()}')
        else:
            payload = f"action:{json.dumps(cmd_data, ensure_ascii=False)}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'Sent action -> {payload.strip()}')

    def action_callback(self, msg):
        """Handle one action command from the legacy topic."""
        try:
            cmd_data = json.loads(msg.data)
            self._send_action_dict(cmd_data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Action command JSON parse failed: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Action command handling failed: {e}")

    def destroy_node(self):
        self.get_logger().info('Closing serial bridge...')
        if hasattr(self, 'bridge'):
            self.bridge.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
