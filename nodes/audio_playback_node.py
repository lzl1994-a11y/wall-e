#!/usr/bin/env python3
"""音频播放节点：订阅 /audio_output → PlaybackService 播放 → USB/I2S 切换

只负责 ROS I/O。播放逻辑在 services/playback_service.py。
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.audio_output import OUTPUT_SAMPLE_RATE
from services.music_protocol import MUSIC_AUDIO_TOPIC
from services.playback_service import PlaybackService


class AudioPlaybackNode(Node):
    def __init__(self):
        super().__init__("audio_playback_node")

        self.declare_parameter("mode", "default")
        self.declare_parameter("sample_rate", OUTPUT_SAMPLE_RATE)

        mode = self.get_parameter("mode").value
        sample_rate = self.get_parameter("sample_rate").value

        self._dialog_state_pub = self.create_publisher(String, "llm_busy", 10)
        self._player = PlaybackService(
            mode=mode,
            sample_rate=sample_rate,
            on_turn_complete=self._on_turn_complete,
        )

        self.create_subscription(UInt8MultiArray, "audio_output", self._on_audio, 10)
        self.create_subscription(UInt8MultiArray, MUSIC_AUDIO_TOPIC, self._on_music_audio, 10)

        self.get_logger().info(
            f"音频播放节点上线 (mode={mode}, sr={sample_rate})"
        )

    def _on_audio(self, msg):
        if not msg.data:
            self._player.mark_turn_end()
            return
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        self._player.play(samples)

    def _on_music_audio(self, msg):
        if not msg.data:
            self._player.mark_stream_end()
            return
        self._player.play(np.frombuffer(bytes(msg.data), dtype=np.int16))

    def _on_turn_complete(self):
        self._dialog_state_pub.publish(String(data="idle"))
        self.get_logger().info("本轮语音播放完成，恢复 ASR 计时")


def main(args=None):
    rclpy.init(args=args)
    node = AudioPlaybackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
