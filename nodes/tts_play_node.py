#!/usr/bin/env python3
"""TTS 播放节点：订阅 tts_text → TTSService 合成 → /audio_output（PCM int16）

只负责 ROS I/O。合成逻辑在 services/tts_service.py。
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.audio_output import OUTPUT_SAMPLE_RATE
from services.tts_pipeline import OrderedTTSPipeline
from services.tts_protocol import decode_turn_end
from services.tts_service import TTSService


class TTSPlayNode(Node):
    def __init__(self):
        super().__init__("tts_play_node")

        self.create_subscription(String, "tts_text", self._on_tts_text, 10)
        self.audio_pub = self.create_publisher(UInt8MultiArray, "audio_output", 10)

        self.declare_parameter("voice", "zh-CN-YunxiaNeural")
        self.declare_parameter("rate", "+20%")
        self.declare_parameter("pitch", "+5Hz")
        self.declare_parameter("sample_rate", OUTPUT_SAMPLE_RATE)
        self.declare_parameter("prefetch_workers", 2)
        self.voice = self.get_parameter("voice").value
        self.rate = self.get_parameter("rate").value
        self.pitch = self.get_parameter("pitch").value
        self.sample_rate = self.get_parameter("sample_rate").value
        self.prefetch_workers = self.get_parameter("prefetch_workers").value

        self._tts = TTSService(
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            sample_rate=self.sample_rate,
        )

        self._pipeline = OrderedTTSPipeline(
            synthesize=self._tts.synthesize,
            on_audio=self._publish_audio,
            on_turn_end=self._publish_turn_end,
            on_error=self._log_synthesis_error,
            workers=self.prefetch_workers,
        )

        self.get_logger().info(
            f"TTS 播放节点上线 "
            f"(voice={self.voice}, rate={self.rate}, pitch={self.pitch}, "
            f"sr={self.sample_rate}, prefetch={self.prefetch_workers})"
        )

    def _on_tts_text(self, msg):
        text = (msg.data or "").strip()
        if not text:
            return
        turn_id = decode_turn_end(text)
        if turn_id is not None:
            self._pipeline.submit_turn_end(turn_id)
            return
        self._pipeline.submit_speech(text)

    def _publish_audio(self, samples, text, elapsed):
        msg = UInt8MultiArray(data=samples.tobytes())
        self.audio_pub.publish(msg)
        self.get_logger().info(
            f"TTS → /audio_output: {len(msg.data)} bytes, "
            f"synthesis={elapsed:.2f}s, text={text[:40]}"
        )

    def _publish_turn_end(self, turn_id):
        self.audio_pub.publish(UInt8MultiArray(data=[]))
        self.get_logger().info(f"TTS 轮次结束: {turn_id}")

    def _log_synthesis_error(self, text, error, elapsed):
        self.get_logger().error(
            f"TTS 合成失败 ({elapsed:.2f}s, text={text[:40]}): {error}"
        )

    def destroy_node(self):
        self._pipeline.shutdown()
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
