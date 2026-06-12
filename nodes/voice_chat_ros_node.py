#!/usr/bin/env python3
"""语音直聊 ROS2 节点：唤醒词 → 语音应答 → VAD → Qwen-Omni → TTS

状态机由 VoiceChatService 驱动，本节点负责 ROS 侧回调：
  on_wake_word   → 播放预合成 WAV + TFT 切聊天页
  on_llm_reply   → TTS 播报 + 屏幕对话框
  on_tool_call   → /action_cmd
  on_llm_timeout → TFT 切待机页 + 日志
"""

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.voice_chat_service import VoiceChatService
from services.tool_dispatcher import build_action_cmd


class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        self.tts_pub = self.create_publisher(String, "tts_text", 10)
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)
        self.action_pub = self.create_publisher(String, "action_cmd", 10)

        self.get_logger().info("正在预热唤醒词 + Qwen-Omni 引擎...")

        self.vc = VoiceChatService()
        self.vc.on_wake_word = self._on_wake_word
        self.vc.on_llm_reply = self._on_llm_reply
        self.vc.on_tool_call = self._on_tool_call
        self.vc.on_llm_timeout = self._on_llm_timeout
        self.vc.start()

        # 唤醒应答 WAV 路径
        root = Path(__file__).resolve().parent.parent
        self._wake_wav = str(root / "assets" / "wake_response.wav")
        self._wake_play_lock = threading.Lock()

        self.get_logger().info("语音直聊节点已上线")

    # ── 唤醒词回调 ──
    def _on_wake_word(self):
        """唤醒词触发：播放预合成语音 + 切 TFT 到聊天页。"""
        self.get_logger().info("唤醒词触发")

        # 切 TFT 到聊天页面
        try:
            screen_msg = String()
            screen_msg.data = json.dumps(
                {"page": "chat", "text": "正在听...", "source": "wake_word"},
                ensure_ascii=False,
            )
            self.dialog_pub.publish(screen_msg)
        except Exception:
            pass

        # 播放预合成应答 WAV（后台线程，不阻塞主循环）
        threading.Thread(target=self._play_wake_response, daemon=True).start()

    def _play_wake_response(self):
        """播放 assets/wake_response.wav。"""
        with self._wake_play_lock:
            if not os.path.exists(self._wake_wav):
                self.get_logger().warn(f"唤醒应答文件不存在: {self._wake_wav}")
                self.get_logger().warn("请先运行 generate_wake_response.py 生成语音文件")
                return

            try:
                import wave
                import sounddevice as sd

                with wave.open(self._wake_wav, "rb") as wf:
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    if n_frames == 0:
                        return
                    audio = wf.readframes(n_frames)

                import numpy as np
                samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(samples, samplerate=sample_rate)
                sd.wait()
                self.get_logger().info("唤醒应答播放完毕")
            except ImportError:
                self.get_logger().error("缺少 sounddevice 库，无法播放唤醒应答")
            except Exception as e:
                self.get_logger().error(f"播放唤醒应答失败: {e}")

    # ── LLM 回调 ──
    def _on_tool_call(self, name, arguments):
        payload = build_action_cmd(name, arguments)
        msg = String()
        msg.data = payload
        self.action_pub.publish(msg)
        self.get_logger().info(f"Tool: {name}({arguments})")

    def _on_llm_reply(self, text):
        text = text.strip()
        if not text:
            return

        # TTS 播报
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f"TTS: {text[:80]}")

        # 屏幕对话框
        dialog = String()
        dialog.data = json.dumps(
            {"text": text, "source": "voice_chat"}, ensure_ascii=False
        )
        self.dialog_pub.publish(dialog)

    # ── 超时回调 ──
    def _on_llm_timeout(self):
        """40s 无 LLM 回复，TFT 切回待机。"""
        self.get_logger().info("LLM 超时，切回待机")
        try:
            screen_msg = String()
            screen_msg.data = json.dumps(
                {"page": "idle", "text": "说「瓦力瓦力」唤醒我", "source": "timeout"},
                ensure_ascii=False,
            )
            self.dialog_pub.publish(screen_msg)
        except Exception:
            pass

    def destroy_node(self):
        self.get_logger().info("正在关闭语音直聊节点...")
        if hasattr(self, "vc"):
            self.vc.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceChatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
