#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 引入底层的听觉血肉引擎
from services.stt_service import STTService
from services.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active
from services.music_protocol import MUSIC_STATE_TOPIC, decode_music_state
from services.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC,
    VAD_SPEECH_ENDED,
    VAD_SPEECH_STARTED,
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
        self._music_active = False
        self._recording_paused = False
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(String, MUSIC_STATE_TOPIC, self._on_music_state, 10)

        # 订阅 LLM 忙闲状态，LLM 处理中暂停 ASR
        self.busy_subscription = self.create_subscription(
            String, 'llm_busy', self._on_llm_busy, 10
        )
        
        self.get_logger().info('⏳ 正在初始化语音识别引擎...')
        
        try:
            # 2. 启动底层引擎，并把“发布消息”的动作作为回调函数塞进去
            self.stt_engine = STTService(
                on_sentence_received=self.on_speech_detected
            )
            self.stt_engine.on_speech_start = self._on_vad_speech_start
            self.stt_engine.on_speech_end = self._on_vad_speech_end
            self.stt_engine.start()
            self.get_logger().info('✅ 听觉节点已上线，正在全天候监听环境声音...')
        except Exception as e:
            self.get_logger().error(f'🔴 底层引擎启动失败: {e}')

    def on_speech_detected(self, text):
        """
        传动轴函数：底层一旦断句成功，立刻触发这里
        """
        if self._game_active or self._llm_busy or self._music_active:
            return
        self.get_logger().info(f'👂 捕捉到人声: "{text}"')
        
        # 将纯文本打包成 ROS 2 的标准 String 消息并广播出去
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)

    def _on_vad_speech_start(self):
        self.dialog_motion_pub.publish(String(data=VAD_SPEECH_STARTED))

    def _on_vad_speech_end(self):
        self.dialog_motion_pub.publish(String(data=VAD_SPEECH_ENDED))

    def _on_llm_busy(self, msg):
        """对话开始时暂停 ASR，AI 语音播放完成后恢复。"""
        if msg.data == "busy":
            self._llm_busy = True
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

    def _on_music_state(self, msg):
        state = decode_music_state(msg.data)
        if state is None:
            return
        active = state["state"] in {"loading", "playing"}
        if active == self._music_active:
            return
        self._music_active = active
        self._sync_recording()

    def _sync_recording(self):
        """Keep the loaded wake model idle while any audio policy blocks recording."""
        paused = self._game_active or self._llm_busy or self._music_active
        if paused == self._recording_paused:
            return
        engine = getattr(self, "stt_engine", None)
        if engine is None:
            return
        self._recording_paused = paused
        if paused:
            self.get_logger().info('🔇 录音暂停，唤醒词模型进入热备')
            engine.pause()
        else:
            self.get_logger().info('🔊 录音恢复，唤醒词检测重新启用')
            engine.resume()

    def destroy_node(self):
        # 节点被杀死时，务必释放底层麦克风资源，防止端口被占死
        self.get_logger().info('🛑 正在安全关闭麦克风...')
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
