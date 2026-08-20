#!/usr/bin/env python3
"""Keyboard text input for exercising the normal LLM/TTS ROS pipeline."""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardSTTNode(Node):
    def __init__(self):
        super().__init__("keyboard_stt_test_node")
        self._publisher = self.create_publisher(String, "voice_text", 10)
        self._stopping = threading.Event()
        self._input_thread = threading.Thread(
            target=self._input_loop,
            name="keyboard-stt-input",
            daemon=True,
        )
        self._input_thread.start()
        self.get_logger().info("键盘输入节点已上线；输入文字并按回车发送。")

    def _publish_text(self, text):
        text = (text or "").strip()
        if not text:
            return False
        self._publisher.publish(String(data=text))
        self.get_logger().info(f"键盘输入: {text}")
        return True

    def _input_loop(self):
        while not self._stopping.is_set() and rclpy.ok():
            try:
                text = input("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            self._publish_text(text)

    def destroy_node(self):
        self._stopping.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardSTTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
