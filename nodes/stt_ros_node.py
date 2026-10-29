#!/usr/bin/env python3
import json
import os
import rclpy
from rclpy.node import Node
try:
    from std_msgs.msg import String, UInt8MultiArray
except ImportError:  # Keep policy-only tests compatible with minimal ROS stubs.
    from std_msgs.msg import String

    class UInt8MultiArray:
        def __init__(self, data=None):
            self.data = [] if data is None else data

# 引入底层的听觉血肉引擎
from services.speech.stt_service import STTService
from services.game.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active
from services.dialog.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC,
    VAD_SPEECH_ENDED,
    VAD_SPEECH_STARTED,
)
from services.remote.remote_protocol import (
    REMOTE_AUDIO_PCM_TOPIC,
    REMOTE_VOICE_STATE_TOPIC,
    ROBOT_AUDIO_PCM_TOPIC,
)
from services.remote.microphone_mode import REMOTE_CONNECTION_TOPIC, RemoteMicrophoneLease
from services.audio.audio_control_protocol import AUDIO_CONTROL_TOPIC, encode_stop_speech_control
from services.audio.startup_prompts import (
    StartupPromptGate, clear_startup_ready, mark_startup_ready,
)

class STTNode(Node):
    def __init__(self):
        super().__init__('walle_ear_node')
        
        # 1. 声明发布者：专往 'voice_text' 这个话题里扔字符串
        self.publisher_ = self.create_publisher(String, 'voice_text', 10)
        self.dialog_motion_pub = self.create_publisher(
            String, DIALOG_MOTION_VAD_TOPIC, 10
        )
        self._game_active = False
        self._llm_busy = False
        self._recording_paused = False
        self._startup_prompt_gate = StartupPromptGate()
        self._startup_utterance_submitted = False
        self._startup_recognition_pub = (
            self.create_publisher(String, "startup_recognition_state", 10)
            if os.environ.get("WALI_STARTUP_STATUS_DIR") else None
        )
        self._shared_capture = os.environ.get("WALI_SHARED_MICROPHONE") == "1"
        self._remote_connected = self._shared_capture
        self._remote_lease = RemoteMicrophoneLease() if self._shared_capture else None
        self._shared_audio_sub = None
        self._audio_control_pub = self.create_publisher(String, AUDIO_CONTROL_TOPIC, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(
            UInt8MultiArray,
            REMOTE_AUDIO_PCM_TOPIC,
            self._on_remote_audio,
            50,
        )
        self.create_subscription(
            String,
            REMOTE_VOICE_STATE_TOPIC,
            self._on_remote_voice_state,
            10,
        )

        # 订阅 LLM 忙闲状态，LLM 处理中暂停 ASR
        self.busy_subscription = self.create_subscription(
            String, 'llm_busy', self._on_llm_busy, 10
        )
        
        self.get_logger().info('⏳ 正在初始化语音识别引擎...')
        
        try:
            # 2. 启动底层引擎，并把“发布消息”的动作作为回调函数塞进去
            self.stt_engine = STTService(
                on_sentence_received=self.on_speech_detected,
                shared_capture=self._shared_capture,
            )
            self.stt_engine.on_speech_start = self._on_vad_speech_start
            self.stt_engine.on_speech_end = self._on_vad_speech_end
            if self._startup_recognition_pub is not None:
                self.stt_engine.on_recognition_state = self._on_recognition_state
                self._publish_startup_recognition_state("idle")
            self.stt_engine.start()
            if self._shared_capture:
                self.stt_engine.set_remote_connected(True)
                self._sync_recording()
                self.create_subscription(String, REMOTE_CONNECTION_TOPIC, self._on_remote_connection, 1)
                self.create_timer(0.5, self._check_remote_connection)
            if os.environ.get("WALI_STARTUP_STATUS_DIR"):
                self.create_timer(0.1, self._sync_recording)
            self._sync_recording()
            mark_startup_ready()
            self.get_logger().info('✅ 听觉节点已上线，正在全天候监听环境声音...')
        except Exception as e:
            self.get_logger().error(f'🔴 底层引擎启动失败: {e}')

    def on_speech_detected(self, text):
        """
        传动轴函数：底层一旦断句成功，立刻触发这里
        """
        if (self._game_active or self._llm_busy or getattr(self, "_remote_connected", False)
                or self._startup_prompt_active()):
            return
        self.get_logger().info(f'👂 捕捉到人声: "{text}"')
        
        # 将纯文本打包成 ROS 2 的标准 String 消息并广播出去
        msg = String()
        msg.data = text
        if getattr(self, "_startup_recognition_pub", None) is not None:
            try:
                subscribers = self.publisher_.get_subscription_count()
            except Exception:
                subscribers = 0
            # A volatile voice_text message with no consumer cannot later
            # produce an LLM busy acknowledgement. Do not hold ready forever.
            self._startup_utterance_submitted = type(subscribers) is int and subscribers > 0
            if self._startup_utterance_submitted:
                self._publish_startup_recognition_state("submitted")
        try:
            self.publisher_.publish(msg)
        except Exception:
            self._startup_utterance_submitted = False
            self._publish_startup_recognition_state("idle")
            raise

    def _publish_startup_recognition_state(self, state):
        publisher = getattr(self, "_startup_recognition_pub", None)
        if publisher is not None:
            publisher.publish(String(data=state))

    def _on_recognition_state(self, busy):
        if busy:
            self._startup_utterance_submitted = False
            state = "busy"
        else:
            state = "submitted" if getattr(self, "_startup_utterance_submitted", False) else "idle"
        self._publish_startup_recognition_state(state)

    def _on_remote_connection(self, message):
        try:
            self._remote_lease.update(json.loads(message.data)["connected"])
        except (TypeError, ValueError, KeyError):
            return
        self._check_remote_connection()

    def _check_remote_connection(self):
        connected = self._remote_lease.remote_active
        if connected == self._remote_connected:
            return
        self._remote_connected = connected
        self.stt_engine.set_remote_connected(connected)
        if connected:
            # The interrupted local turn no longer owns the busy state.
            self._llm_busy = False
            self._audio_control_pub.publish(String(data=encode_stop_speech_control("remote_connection")))
        self._sync_recording()
        self.get_logger().info(
            "远程已连接，麦克风用于对讲，本地唤醒暂停" if connected
            else "远程已断开，恢复本地唤醒和 ASR → LLM 对话"
        )

    def _set_shared_microphone_input(self, enabled):
        if not enabled and self._shared_audio_sub is not None:
            self.destroy_subscription(self._shared_audio_sub)
            self._shared_audio_sub = None
        elif enabled and self._shared_audio_sub is None:
            from rclpy.qos import qos_profile_sensor_data

            self._shared_audio_sub = self.create_subscription(
                UInt8MultiArray, ROBOT_AUDIO_PCM_TOPIC,
                lambda msg: self.stt_engine.accept_shared_pcm(bytes(msg.data)),
                qos_profile_sensor_data,
            )

    def _on_vad_speech_start(self):
        self.dialog_motion_pub.publish(String(data=VAD_SPEECH_STARTED))

    def _on_vad_speech_end(self):
        self.dialog_motion_pub.publish(String(data=VAD_SPEECH_ENDED))

    def _on_llm_busy(self, msg):
        """对话开始时暂停 ASR，AI 语音播放完成后恢复。"""
        if msg.data == "busy":
            self._llm_busy = True
            self._startup_utterance_submitted = False
        elif msg.data == "idle":
            self._llm_busy = False
        else:
            return
        self._sync_recording()

    def _on_game_state(self, msg):
        active = game_is_active(msg.data)
        if active == self._game_active:
            return
        self._game_active = active
        self._sync_recording()

    def _on_remote_voice_state(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            return
        state = payload.get("state") if isinstance(payload, dict) else None
        mode = payload.get("mode") if isinstance(payload, dict) else None
        engine = getattr(self, "stt_engine", None)
        if engine is None:
            return
        if state == "start":
            accepted = (
                engine.begin_remote_session()
                if mode == "session"
                else engine.begin_remote_turn()
            )
            if not accepted:
                self.get_logger().warning("当前语音状态不允许开始远程对话")
        elif state == "stop":
            if mode == "session":
                engine.end_remote_session()
            else:
                engine.end_remote_turn()

    def _on_remote_audio(self, msg):
        engine = getattr(self, "stt_engine", None)
        if engine is not None:
            engine.accept_remote_audio(bytes(msg.data or b""), sample_rate=16000)

    def _sync_recording(self):
        """Keep the loaded wake model idle while any audio policy blocks recording."""
        paused = (self._game_active or self._llm_busy or getattr(self, "_remote_connected", False)
                  or self._startup_prompt_active())
        if paused == self._recording_paused:
            return
        engine = getattr(self, "stt_engine", None)
        if engine is None:
            return
        self._recording_paused = paused
        if getattr(self, "_shared_capture", False):
            self._set_shared_microphone_input(not paused)
        if paused:
            self.get_logger().info('🔇 录音暂停，唤醒词模型进入热备')
            engine.pause()
        else:
            self.get_logger().info('🔊 录音恢复，唤醒词检测重新启用')
            engine.resume()

    def _startup_prompt_active(self):
        gate = getattr(self, "_startup_prompt_gate", None)
        return gate is not None and gate.active()

    def destroy_node(self):
        # 节点被杀死时，务必释放底层麦克风资源，防止端口被占死
        self.get_logger().info('🛑 正在安全关闭麦克风...')
        clear_startup_ready()
        if hasattr(self, 'stt_engine'):
            self.stt_engine.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
