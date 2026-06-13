#!/usr/bin/env python3
"""TTS 播放节点：订阅 tts_text → TTSService 合成 → /audio_output（PCM int16）

只负责 ROS I/O。合成逻辑在 services/tts_service.py。
"""

import sys
import threading
import queue
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.tts_service import TTSService


class TTSPlayNode(Node):
    def __init__(self):
        super().__init__("tts_play_node")

        self.create_subscription(String, "tts_text", self._on_tts_text, 10)
        self.audio_pub = self.create_publisher(UInt8MultiArray, "audio_output", 10)

        self.declare_parameter("voice", "zh-CN-YunxiaNeural")
        self.declare_parameter("rate", "+20%")
        self.declare_parameter("pitch", "+5Hz")
        self.voice = self.get_parameter("voice").value
        self.rate = self.get_parameter("rate").value
        self.pitch = self.get_parameter("pitch").value

        self._tts = TTSService(voice=self.voice, rate=self.rate, pitch=self.pitch)

        self._text_queue = queue.Queue()
        self._worker = threading.Thread(target=self._synthesis_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"TTS 播放节点上线 (voice={self.voice}, rate={self.rate}, pitch={self.pitch})"
        )

    def _on_tts_text(self, msg):
        text = (msg.data or "").strip()
        if not text:
            return
        self._text_queue.put(text)

    def _synthesis_worker(self):
        """工作线程：顺序取文本 → 合成 → publish。"""
        while True:
            text = self._text_queue.get()
            try:
                samples = self._tts.synthesize(text)
                msg = UInt8MultiArray(data=samples.tobytes())
                self.audio_pub.publish(msg)
                self.get_logger().info(
                    f"TTS → /audio_output: {len(msg.data)} bytes, text={text[:40]}"
                )
            except Exception as e:
                self.get_logger().error(f"TTS 合成失败: {e}")

    def destroy_node(self):
        self._tts.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TTSPlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
