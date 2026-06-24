#!/usr/bin/env python3
"""串口通信节点：唯一的串口持有者。

订阅 Topic 并透传下位机：
  /screen_dialog → 屏幕文字（you: / ai:）
  /tft_cmd       → TFT 控制指令（eyeaction:...）
  /pca9685_raw   → PCA9685 15 通道原始值（由 hardware_bridge_node 产出）

动作分发由 action_ros_node 负责，硬件驱动由 hardware_bridge_node 负责。
本节点只做串口透传，不做任何业务解析。
"""

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

        # 订阅 Topic
        self.create_subscription(String, 'screen_dialog', self.screen_dialog_callback, 10)
        self.create_subscription(String, 'tft_cmd', self.tft_cmd_callback, 10)
        self.create_subscription(String, 'pca9685_raw', self.pca9685_callback, 10)

        self.get_logger().info('Serial ROS node is online (sole serial owner).')

    # ------------------------------------------------------------------
    # screen_dialog: 屏幕文字
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # tft_cmd: 表情控制指令（原 action_ros_node 直接写串口）
    # ------------------------------------------------------------------
    def tft_cmd_callback(self, msg):
        if self.bridge.send_raw(msg.data):
            self.get_logger().debug(f'[Serial] TFT cmd forwarded: {msg.data.strip()}')

    # ------------------------------------------------------------------
    # pca9685_raw: 硬件 15 通道原始值（原 hardware_bridge_node 直接写串口）
    # ------------------------------------------------------------------
    def pca9685_callback(self, msg):
        payload = msg.data + '\n'
        if self.bridge.send_raw(payload):
            self.get_logger().debug(f'[Serial] PCA9685 forwarded ({len(msg.data)} bytes)')

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
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
