#!/usr/bin/env python3
"""音频播放节点：订阅 /audio_output → PlaybackService 播放 → USB/I2S 切换

只负责 ROS I/O。播放逻辑在 services/playback_service.py。
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.playback_service import PlaybackService


class AudioPlaybackNode(Node):
    def __init__(self):
        super().__init__("audio_playback_node")

        self.declare_parameter("mode", "default")
        self.declare_parameter("sample_rate", 16000)

        mode = self.get_parameter("mode").value
        sample_rate = self.get_parameter("sample_rate").value

        self._player = PlaybackService(mode=mode, sample_rate=sample_rate)

        self.create_subscription(UInt8MultiArray, "audio_output", self._on_audio, 10)

        self.get_logger().info(
            f"音频播放节点上线 (mode={mode}, sr={sample_rate})"
        )

    def _on_audio(self, msg):
        if not msg.data:
            return
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        self._player.play(samples)


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
