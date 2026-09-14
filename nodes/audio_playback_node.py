#!/usr/bin/env python3
"""音频播放节点：统一混合对话、系统提示和音乐，再输出到 USB/I2S。

只负责 ROS I/O。播放与降音逻辑在 services/mixing_playback_service.py。
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.audio_output import OUTPUT_SAMPLE_RATE
from services.music_protocol import MUSIC_AUDIO_TOPIC
from services.mixing_playback_service import MixingPlaybackService
from services.esp32_network_prompt import (
    ESP32_NETCFG_STATUS_TOPIC,
    Esp32NetworkPromptSelector,
)
from services.system_audio_protocol import (
    SYSTEM_AUDIO_DONE_TOPIC,
    SYSTEM_AUDIO_TOPIC,
    decode_system_audio,
)
from services.wake_audio_protocol import (
    WAKE_AUDIO_DONE_TOPIC,
    WAKE_AUDIO_TOPIC,
    decode_wake_audio,
)


class AudioPlaybackNode(Node):
    _TTS_SEQUENCE_LABEL_PREFIX = "walle.tts_pcm_sequence:"

    def __init__(self):
        super().__init__("audio_playback_node")

        self.declare_parameter("mode", "default")
        self.declare_parameter("sample_rate", OUTPUT_SAMPLE_RATE)

        mode = self.get_parameter("mode").value
        sample_rate = self.get_parameter("sample_rate").value

        self._dialog_state_pub = self.create_publisher(String, "llm_busy", 10)
        self._wake_done_pub = self.create_publisher(String, WAKE_AUDIO_DONE_TOPIC, 10)
        self._system_done_pub = self.create_publisher(String, SYSTEM_AUDIO_DONE_TOPIC, 10)
        self._player = MixingPlaybackService(
            mode=mode,
            sample_rate=sample_rate,
            on_turn_complete=self._on_turn_complete,
            on_wake_complete=self._on_wake_complete,
            on_system_complete=self._on_system_complete,
        )
        self._last_tts_sequence = 0
        self._received_tts_chunks = 0
        self._missing_tts_chunks = 0
        self._network_prompts = None
        try:
            self._network_prompts = Esp32NetworkPromptSelector(
                Path(__file__).resolve().parent.parent / "assets"
            )
        except (OSError, ValueError) as exc:
            self.get_logger().warning(f"ESP32 配网提示音不可用: {exc}")

        self.create_subscription(
            UInt8MultiArray,
            "audio_output",
            self._on_audio,
            QoSProfile(depth=128, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(UInt8MultiArray, MUSIC_AUDIO_TOPIC, self._on_music_audio, 10)
        self.create_subscription(String, WAKE_AUDIO_TOPIC, self._on_wake_audio, 10)
        self.create_subscription(String, SYSTEM_AUDIO_TOPIC, self._on_system_audio, 10)
        # A short-lived retained history closes the process-start race. The
        # selector rejects timestamped stale events if this node restarts later.
        self.create_subscription(
            String,
            ESP32_NETCFG_STATUS_TOPIC,
            self._on_netcfg_status,
            QoSProfile(
                depth=4,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        self.get_logger().info(
            f"音频播放节点上线 (mode={mode}, sr={sample_rate})"
        )

    def _on_audio(self, msg):
        if not msg.data:
            self._log_tts_turn_diagnostics()
            self._player.mark_turn_end()
            return
        self._track_tts_sequence(msg)
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        self._player.play(samples)

    def _track_tts_sequence(self, msg):
        sequence = 0
        for dimension in msg.layout.dim:
            if dimension.label.startswith(self._TTS_SEQUENCE_LABEL_PREFIX):
                try:
                    sequence = int(
                        dimension.label[len(self._TTS_SEQUENCE_LABEL_PREFIX):]
                    )
                except ValueError:
                    self.get_logger().warning(
                        f"Invalid TTS PCM sequence metadata: {dimension.label}"
                    )
                break
        # Other publishers (for example FC audio) use the same topic but do
        # not carry this diagnostic marker.
        if sequence <= 0:
            return

        if self._last_tts_sequence:
            expected = self._last_tts_sequence + 1
            if sequence > expected:
                missing = sequence - expected
                self._missing_tts_chunks += missing
                self.get_logger().warning(
                    "TTS PCM sequence gap: expected=%d received=%d "
                    "missing=%d" % (expected, sequence, missing)
                )
            elif sequence < expected:
                # A TTS node restart restarts its local counter. Treat it as
                # a new diagnostic stream rather than reporting a false gap.
                self.get_logger().info(
                    "TTS PCM sequence restarted: previous=%d received=%d"
                    % (self._last_tts_sequence, sequence)
                )
                self._missing_tts_chunks = 0
        elif sequence != 1:
            self.get_logger().warning(
                "TTS PCM sequence started at %d; earlier chunks were sent "
                "before this playback node began receiving" % sequence
            )

        self._last_tts_sequence = sequence
        self._received_tts_chunks += 1

    def _log_tts_turn_diagnostics(self):
        if not self._received_tts_chunks:
            return
        self.get_logger().info(
            "TTS PCM diagnostics: received_chunks=%d missing_chunks=%d "
            "last_sequence=%d" % (
                self._received_tts_chunks,
                self._missing_tts_chunks,
                self._last_tts_sequence,
            )
        )
        self._last_tts_sequence = 0
        self._received_tts_chunks = 0
        self._missing_tts_chunks = 0

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

    def _on_system_audio(self, msg):
        try:
            request_id, _cue, pcm = decode_system_audio(msg.data)
        except ValueError as exc:
            self.get_logger().warning(f"系统提示音格式错误: {exc}")
            return
        self._player.play_prompt(
            np.frombuffer(pcm, dtype=np.int16), "system", request_id
        )

    def _on_netcfg_status(self, msg):
        if self._network_prompts is None:
            return
        selected = self._network_prompts.select(msg.data)
        if selected is None:
            return
        request_id, cue, pcm = selected
        self.get_logger().info(f"播放系统提示音: {cue}")
        self._player.play_prompt(
            np.frombuffer(pcm, dtype=np.int16), "system", request_id
        )

    def _on_system_complete(self, request_id):
        self._system_done_pub.publish(String(data=request_id))

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
