#!/usr/bin/env python3
"""Keyboard-driven STT test node.

Type text in the terminal and this node publishes it to the same `voice_text`
topic used by `stt_ros_node.py`, so the LLM pipeline can be tested without a
microphone.
"""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardSTTNode(Node):
    def __init__(self):
        super().__init__('keyboard_stt_test_node')
        self.publisher_ = self.create_publisher(String, 'voice_text', 10)
        self._running = True
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()
        self.get_logger().info('Keyboard STT test node started. Type text and press Enter.')
        self.get_logger().info('Type /exit or press Ctrl+C to quit.')

    def _input_loop(self):
        while self._running and rclpy.ok():
            try:
                text = input('voice_text> ').strip()
            except (EOFError, KeyboardInterrupt):
                self._running = False
                rclpy.shutdown()
                break

            if not text:
                continue

            if text.lower() in {'/exit', 'exit', 'quit', '/quit'}:
                self._running = False
                rclpy.shutdown()
                break

            msg = String()
            msg.data = text
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published voice_text: {text}')

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardSTTNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == '__main__':
    main()
