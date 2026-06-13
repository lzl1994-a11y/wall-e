#!/usr/bin/env python3
"""语音直聊 ROS2 节点：唤醒词 → 语音应答 → VAD → Qwen-Omni → TTS

状态机由 VoiceChatService 驱动，本节点负责 ROS 侧回调：
  on_wake_word   → 播放预合成 WAV + TFT 切聊天页
  on_llm_chunk   → 流式文本块，2 标点攒一句 → tts_text
  on_llm_reply   → 最终完整回复 → screen_dialog
  on_tool_call   → /action_cmd
  on_llm_timeout → TFT 切待机页 + 日志
"""

import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.voice_chat_service import VoiceChatService
from services.tool_dispatcher import build_action_cmd

# 去掉 TTS 不需要的符号（保留中文标点和空格）
TTS_CLEAN_RE = re.compile(r'[*#_~`>\[\]\(\)\{\}]')

class VoiceChatNode(Node):
    def __init__(self):
        super().__init__("voice_chat_node")

        self.tts_pub = self.create_publisher(String, "tts_text", 10)
        self.dialog_pub = self.create_publisher(String, "screen_dialog", 10)
        self.action_pub = self.create_publisher(String, "action_cmd", 10)

        self.get_logger().info("正在预热唤醒词 + Qwen-Omni 引擎...")

        self.vc = VoiceChatService()
        self.vc.on_wake_word = self._on_wake_word
        self.vc.on_llm_chunk = self._on_llm_chunk
        self.vc.on_llm_reply = self._on_llm_reply
        self.vc.on_tool_call = self._on_tool_call
        self.vc.on_llm_timeout = self._on_llm_timeout
        self.vc.start()

        # 流式 TTS 状态
        self._sentence_buffer = ""     # 当前攒的句子
        self._punc_count = 0           # 标点计数
        self._correction_done = False  # 第一行纠错已提取
        self.punctuations = {"。", "？", ".", "?", "！", "!"}

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

    def _on_llm_chunk(self, text):
        """流式文本块：跳过纠错首行，2 标点攒一句 → tts_text。"""
        if not text:
            return

        self._sentence_buffer += text

        # 跳过第一行（纠错文本前缀）
        if not self._correction_done:
            if "\n" in self._sentence_buffer:
                parts = self._sentence_buffer.split("\n", 1)
                self._sentence_buffer = parts[1] if len(parts) > 1 else ""
                self._correction_done = True
                # 检查新 buffer 里是否已有标点
                self._punc_count = sum(
                    1 for c in self._sentence_buffer if c in self.punctuations
                )
            elif len(self._sentence_buffer) > 60:
                self._correction_done = True
                self._punc_count = sum(
                    1 for c in self._sentence_buffer if c in self.punctuations
                )
            else:
                return

        # 按标点累积
        for char in text:
            if char in self.punctuations:
                self._punc_count += 1

        if self._punc_count >= 2:
            clean = self._sentence_buffer.strip()
            tts_safe = TTS_CLEAN_RE.sub("", clean)
            if tts_safe.strip():
                msg = String()
                msg.data = tts_safe.strip()
                self.tts_pub.publish(msg)
                self.get_logger().info(f"TTS: {tts_safe.strip()[:80]}")
            self._sentence_buffer = ""
            self._punc_count = 0

    def _on_llm_reply(self, text):
        """最终完整回复 → 解析 you/ai → screen_dialog。"""
        text = text.strip()
        if not text:
            return

        # 解析 you: / ai: 格式
        corrected_text = ""
        ai_text = text
        if text.startswith("you:"):
            lines = text.split("\n", 1)
            corrected_text = lines[0][4:].strip()
            ai_text = lines[1].strip() if len(lines) > 1 else ""
            if ai_text.startswith("ai:"):
                ai_text = ai_text[3:].strip()

        # flush 残留 TTS 文本
        if self._sentence_buffer.strip():
            tts_safe = TTS_CLEAN_RE.sub("", self._sentence_buffer.strip())
            if tts_safe.strip():
                msg = String()
                msg.data = tts_safe.strip()
                self.tts_pub.publish(msg)
                self.get_logger().info(f"TTS tail: {tts_safe.strip()[:80]}")

        # 屏幕对话框（对齐 llm_ros_node 格式）
        dialog = String()
        dialog.data = json.dumps({
            "turn_id": uuid.uuid4().hex[:12],
            "corrected_text": corrected_text,
            "ai_text": ai_text,
            "actions": [],
            "source": "voice_chat",
        }, ensure_ascii=False)
        self.dialog_pub.publish(dialog)
        self.get_logger().info(f"Screen: {ai_text[:60]}")

        # 重置流式状态
        self._sentence_buffer = ""
        self._punc_count = 0
        self._correction_done = False

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
