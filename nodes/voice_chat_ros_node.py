#!/usr/bin/env python3
"""语音直聊 ROS2 节点：VoiceChatService → tts_text 话题

替代 stt_ros_node + llm_ros_node 的串行链路。
音频直送 Qwen-Omni，回复文本直接发布给 TTS。
"""

import json
import sys
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from services.voice_chat_service import VoiceChatService


class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        self.tts_pub = self.create_publisher(String, "tts_text", 10)
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)

        self.get_logger().info("正在预热 Qwen-Omni 直聊引擎...")

        self.vc = VoiceChatService()
        self.vc.on_llm_reply = self._on_llm_reply
        self.vc.start()
        self.get_logger().info("语音直聊节点已上线（无回声防护，后续补）")

    def _on_llm_reply(self, text):
        text = text.strip()
        if not text:
            return

        # TTS 播报
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f"TTS: {text[:80]}")

        # 屏幕对话框
        dialog = String()
        dialog.data = json.dumps(
            {"text": text, "source": "voice_chat"}, ensure_ascii=False
        )
        self.dialog_pub.publish(dialog)

    def destroy_node(self):
        self.get_logger().info("正在关闭语音直聊节点...")
        if hasattr(self, "vc"):
            self.vc.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceChatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()