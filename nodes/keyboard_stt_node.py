#!/usr/bin/env python3
"""Keyboard-driven STT test node.

Type text in the terminal and this node publishes it to the same `voice_text`
topic used by `stt_ros_node.py`, so the LLM pipeline can be tested without a
microphone.
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.impl.rcutils_logger import RcutilsLogger
from std_msgs.msg import String


class KeyboardSTTNode(Node):
    def __init__(self):
        super().__init__('keyboard_stt_test_node')
        self.publisher_ = self.create_publisher(String, 'voice_text', 10)
        self.get_logger().info('Keyboard STT test node started. Type text and press Enter.')
        self.get_logger().info('Type /exit or press Ctrl+C to quit.')

    def read_line(self):
        """Read terminal input with UTF-8/GB18030 fallback for serial/SSH consoles."""
        sys.stdout.write('voice_text> ')
        sys.stdout.flush()

        raw = sys.stdin.buffer.readline()
        if raw == b'':
            raise EOFError

        for encoding in ('utf-8', 'gb18030', 'gbk'):
            try:
                return raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw.decode('utf-8', errors='replace').strip()

    def publish_text(self, text):
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published voice_text: {text}')


def safe_shutdown():
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardSTTNode()

    try:
        while rclpy.ok():
            try:
                text = node.read_line()
            except EOFError:
                break

            if not text:
                continue

            if text.lower() in {'/exit', 'exit', 'quit', '/quit'}:
                break

            node.publish_text(text)
            rclpy.spin_once(node, timeout_sec=0.0)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        safe_shutdown()


if __name__ == '__main__':
    main()
