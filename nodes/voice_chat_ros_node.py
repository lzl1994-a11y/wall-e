#!/usr/bin/env python3
"""语音直聊 ROS2 节点：VoiceChatService → tts_text 话题

替代 stt_ros_node + llm_ros_node 的串行链路。
音视频直送 Qwen-Omni，回复文本直接发布给 TTS。
"""

import queue
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.voice_chat_service import VoiceChatService


class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        # 输出：TTS 播报
        self.tts_pub = self.create_publisher(String, "tts_text", 10)

        # 输出：屏幕对话气泡（兼容 llm_ros_node 格式）
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)

        # 输入：TTS 状态反馈（播完通知）
        self.tts_done_sub = self.create_subscription(
            String, "tts_done", self._on_tts_done, 10
        )

        self.get_logger().info("正在预热 Qwen-Omni 直聊引擎...")

        self.vc = VoiceChatService()
        # 每次 LLM 回复都发两条：TTS 播报 + 屏幕显示
        self.vc.on_llm_reply = self._on_llm_reply

        # 防回声：TTS 播放时暂停麦克风
        self._tts_busy = False
        self._pending_replies = queue.Queue()

        self.vc.start()
        self.get_logger().info("语音直聊节点已上线")

    def _on_llm_reply(self, text):
        """VoiceChatService 回调：发 TTS，TTS 忙时排队"""
        payload = {"text": text.strip()}
        if not payload["text"]:
            return

        if self._tts_busy:
            self._pending_replies.put(payload)
            self.get_logger().info(f"TTS 忙，排队: {payload['text'][:30]}...")
            return

        self._send_to_tts(payload)

    def _send_to_tts(self, payload):
        text = payload["text"]
        self._tts_busy = True
        self.vc.pause()

        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f"TTS: {text[:50]}...")

        # 屏幕对话框
        import json
        dialog = String()
        dialog.data = json.dumps(
            {"text": text, "source": "voice_chat"}, ensure_ascii=False
        )
        self.dialog_pub.publish(dialog)

    def _on_tts_done(self, msg):
        """TTS 播完，发下一条或恢复监听"""
        if self._pending_replies.qsize():
            try:
                next_payload = self._pending_replies.get_nowait()
                self._send_to_tts(next_payload)
                return
            except queue.Empty:
                pass

        self._tts_busy = False
        self.vc.resume()

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