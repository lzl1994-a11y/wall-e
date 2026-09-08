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
from services.mixing_playback_service import MixingPlaybackService
from services.wake_audio_protocol import (
    WAKE_AUDIO_DONE_TOPIC,
    WAKE_AUDIO_TOPIC,
    decode_wake_audio,
)


class AudioPlaybackNode(Node):
    def __init__(self):
        super().__init__("audio_playback_node")

        self.declare_parameter("mode", "default")
        self.declare_parameter("sample_rate", OUTPUT_SAMPLE_RATE)

        mode = self.get_parameter("mode").value
        sample_rate = self.get_parameter("sample_rate").value

        self._dialog_state_pub = self.create_publisher(String, "llm_busy", 10)
        self._wake_done_pub = self.create_publisher(String, WAKE_AUDIO_DONE_TOPIC, 10)
        self._player = MixingPlaybackService(
            mode=mode,
            sample_rate=sample_rate,
            on_turn_complete=self._on_turn_complete,
            on_wake_complete=self._on_wake_complete,
        )

        self.create_subscription(UInt8MultiArray, "audio_output", self._on_audio, 10)
        self.create_subscription(UInt8MultiArray, MUSIC_AUDIO_TOPIC, self._on_music_audio, 10)
        self.create_subscription(String, WAKE_AUDIO_TOPIC, self._on_wake_audio, 10)

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
        self._player.play_music(np.frombuffer(bytes(msg.data), dtype=np.int16))

    def _on_turn_complete(self):
        self._dialog_state_pub.publish(String(data="idle"))
        self.get_logger().info("本轮语音播放完成，恢复 ASR 计时")

    def _on_wake_audio(self, msg):
        try:
            request_id, pcm = decode_wake_audio(msg.data)
        except ValueError as exc:
            self.get_logger().warning(f"唤醒音频格式错误: {exc}")
            return
        self._player.play_wake(np.frombuffer(pcm, dtype=np.int16), request_id)

    def _on_wake_complete(self, request_id):
        self._wake_done_pub.publish(String(data=request_id))

    def destroy_node(self):
        self._player.close()
        return super().destroy_node()


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
