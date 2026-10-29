#!/usr/bin/env python3
"""音频播放节点：统一混合对话、系统提示和音乐，再输出到 USB/I2S。

只负责 ROS I/O。播放与降音逻辑在 services/audio/mixing_playback_service.py。
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt8MultiArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.audio.audio_output import OUTPUT_SAMPLE_RATE
from services.audio.audio_control_protocol import (
    AUDIO_CONTROL_TOPIC,
    decode_audio_control,
)
from services.audio.music_protocol import MUSIC_AUDIO_TOPIC
from services.audio.mixing_playback_service import MixingPlaybackService
from services.audio.startup_prompts import (
    StartupPromptController, mark_startup_ready, clear_startup_ready,
)
from services.dialog.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC, VAD_SPEECH_STARTED, VAD_SPEECH_ENDED,
)
from services.audio.esp32_network_prompt import (
    ESP32_NETCFG_STATUS_TOPIC,
    Esp32NetworkPromptSelector,
)
from services.audio.system_audio_protocol import (
    SYSTEM_AUDIO_DONE_TOPIC,
    SYSTEM_AUDIO_TOPIC,
    decode_system_audio,
)
from services.audio.wake_audio_protocol import (
    WAKE_AUDIO_DONE_TOPIC,
    WAKE_AUDIO_TOPIC,
    decode_wake_audio,
)
from services.remote.remote_protocol import (
    REMOTE_AUDIO_PLAYBACK_TOPIC,
    REMOTE_INTERCOM_STATE_TOPIC,
)
from services.remote.audio_playback_gate import AudioPlaybackGate, audio_message_epoch


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
        self._remote_audio_gate = AudioPlaybackGate()
        self._received_tts_chunks = 0
        self._missing_tts_chunks = 0
        self._startup_prompts = None
        self._startup_timer = None
        self._dialogue_busy = False
        self._startup_input_pending = False
        self._startup_recognition_busy = False
        self._startup_submission_pending = False
        try:
            self._startup_prompts = StartupPromptController()
        except (OSError, ValueError) as exc:
            self.get_logger().warning(f"启动提示音不可用: {exc}")
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
        self.create_subscription(
            UInt8MultiArray,
            REMOTE_AUDIO_PLAYBACK_TOPIC,
            self._on_remote_audio,
            QoSProfile(depth=4, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            String,
            REMOTE_INTERCOM_STATE_TOPIC,
            self._on_remote_intercom_state,
            10,
        )
        self.create_subscription(
            String, AUDIO_CONTROL_TOPIC, self._on_audio_control, 10
        )
        self.create_subscription(UInt8MultiArray, MUSIC_AUDIO_TOPIC, self._on_music_audio, 10)
        self.create_subscription(String, WAKE_AUDIO_TOPIC, self._on_wake_audio, 10)
        self.create_subscription(String, SYSTEM_AUDIO_TOPIC, self._on_system_audio, 10)
        self.create_subscription(String, "llm_busy", self._on_dialogue_state, 10)
        if self._startup_prompts is not None and self._startup_prompts.enabled:
            self.create_subscription(String, DIALOG_MOTION_VAD_TOPIC, self._on_startup_vad, 10)
            self.create_subscription(String, "startup_recognition_state", self._on_startup_recognition_state, 10)
            self._startup_timer = self.create_timer(0.1, self._poll_startup_prompts)
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
        mark_startup_ready()

    def _on_dialogue_state(self, msg):
        if msg.data in {"busy", "idle"}:
            self._dialogue_busy = msg.data == "busy"
            if self._dialogue_busy:
                self._startup_submission_pending = False
                self._startup_input_pending = getattr(self, "_startup_recognition_busy", False)
                self._interrupt_startup_prompt()

    def _on_startup_recognition_state(self, msg):
        if msg.data == "busy":
            self._startup_recognition_busy = True
            self._startup_submission_pending = False
            self._startup_input_pending = True
        elif msg.data == "submitted":
            self._startup_recognition_busy = False
            self._startup_submission_pending = not self._dialogue_busy
            self._startup_input_pending = self._startup_submission_pending
        elif msg.data == "idle":
            self._startup_recognition_busy = False
            self._startup_input_pending = getattr(self, "_startup_submission_pending", False)

    def _on_startup_vad(self, msg):
        if msg.data == VAD_SPEECH_STARTED:
            self._startup_input_pending = True
            self._interrupt_startup_prompt()
            controller = getattr(self, "_startup_prompts", None)
            if controller is not None and "voice_chat" in controller.required:
                # Direct audio has its own turn state machine. An established
                # user turn takes priority over optional startup feedback.
                controller.skip_ready()
        elif msg.data == VAD_SPEECH_ENDED:
            # ASR owns completion. Fixed delays can interrupt long recognition
            # or its fallback; an unknown outcome delays only the optional cue.
            pass

    def _interrupt_startup_prompt(self):
        controller = getattr(self, "_startup_prompts", None)
        request_id = controller.pending_request if controller is not None else None
        if request_id:
            self._player.stop_speech()
            controller.cancel(request_id)

    def _poll_startup_prompts(self):
        controller = self._startup_prompts
        if controller is None:
            return
        if controller.finished:
            timer = getattr(self, "_startup_timer", None)
            if timer is not None:
                timer.cancel()
                self._startup_timer = None
            return
        if (controller.directory / "stopping").exists():
            self._interrupt_startup_prompt()
            return
        if (self._dialogue_busy or self._remote_audio_gate.active
                or getattr(self, "_startup_input_pending", False)
                or self._player.foreground_active):
            return
        try:
            prompt = controller.next_prompt()
            if prompt is not None:
                request_id, cue, pcm = prompt
                self.get_logger().info(f"播放启动提示音: {cue}")
                self._player.play_prompt(np.frombuffer(pcm, dtype=np.int16), "system", request_id)
        except (OSError, ValueError) as exc:
            self.get_logger().warning(f"启动提示音播放准备失败: {exc}")

    def _on_audio(self, msg):
        if not msg.data:
            self._log_tts_turn_diagnostics()
            self._player.mark_turn_end()
            return
        self._interrupt_startup_prompt()
        self._track_tts_sequence(msg)
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        self._player.play(samples)

    def _on_audio_control(self, msg):
        control = decode_audio_control(msg.data)
        if control is None:
            return
        self._interrupt_startup_prompt()
        if control["source"] in {"remote_safety", "remote_audio_track_ended"}:
            self._remote_audio_gate.close()
        self._player.stop_speech()
        self.get_logger().info(
            f"前景音频已停止 (source={control['source']})"
        )

    def _on_remote_audio(self, msg):
        if (not msg.data or len(msg.data) % 2
                or not self._remote_audio_gate.accepts(audio_message_epoch(msg))):
            return
        self._interrupt_startup_prompt()
        self._player.play_realtime(
            np.frombuffer(bytes(msg.data), dtype=np.int16)
        )

    def _on_remote_intercom_state(self, msg):
        changed = self._remote_audio_gate.update(msg.data)
        if changed:
            self._interrupt_startup_prompt()
        if changed and not self._remote_audio_gate.active:
            self._player.stop_speech()
            self.get_logger().info("实时对讲接收已停止")

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
        self._interrupt_startup_prompt()
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
        controller = getattr(self, "_startup_prompts", None)
        if controller is not None:
            if controller.finish(request_id):
                cue = "startup_ready" if request_id.startswith("startup_ready-") else "startup_begin"
                self.get_logger().info(f"启动提示音队列完成: {cue}")
        self._system_done_pub.publish(String(data=request_id))

    def destroy_node(self):
        clear_startup_ready()
        self._interrupt_startup_prompt()
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
