#!/usr/bin/env python3
"""Keyboard-driven STT test node.

Type text in the terminal and this node publishes it to the same `voice_text`
topic used by `stt_ros_node.py`, so the LLM pipeline can be tested without a
microphone.
"""

import sys
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

    def _read_line(self):
        """Read terminal input with UTF-8/GB18030 fallback for serial/SSH consoles."""
        sys.stdout.write('voice_text> ')
        sys.stdout.flush()

        raw = sys.stdin.buffer.readline()
        if raw == b'':
            raise EOFError

        for encoding in ('utf-8', 'gb18030', 'gbk'):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode('utf-8', errors='replace')

    def _input_loop(self):
        while self._running and rclpy.ok():
            try:
                text = self._read_line().strip()
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
