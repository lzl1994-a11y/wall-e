#!/usr/bin/env python3
"""Capture the robot microphone for the WebRTC live intercom.

This node deliberately does not run wake-word detection, ASR, LLM, TTS, or
voice-history storage. ``AudioPipeline`` is reused only for device discovery,
APM, and conversion to the 16 kHz mono PCM contract consumed by the WebRTC
gateway.
"""

from __future__ import annotations

from array import array
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.audio.audio_pipeline import AudioPipeline
from services.remote.remote_protocol import ROBOT_AUDIO_PCM_TOPIC


class RemoteIntercomNode(Node):
    def __init__(self):
        super().__init__("remote_intercom_node")
        self._audio_pub = self.create_publisher(
            UInt8MultiArray,
            ROBOT_AUDIO_PCM_TOPIC,
            qos_profile_sensor_data,
        )
        self._capture = AudioPipeline(raw_only=True)
        self._capture.on_raw_pcm = self._publish_audio
        self._capture.start()
        self.get_logger().info(
            "实时对讲音频采集已上线（仅转发 PCM，不经过 ASR/LLM、不保存）"
        )

    def _publish_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        message = UInt8MultiArray()
        message.data = array("B", pcm)
        self._audio_pub.publish(message)

    def destroy_node(self):
        self._capture.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RemoteIntercomNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
