"""直接语音对话服务：AudioPipeline(采集+唤醒词+VAD) → state machine → multimodal LLM → TTS

状态机:
  IDLE        ── 待机，等待唤醒词
  AWAKE       ── 已唤醒，VAD 监听语音
  LLM_PENDING ── 语音已发送，等待 LLM 回复

唤醒词触发 → 播放预合成语音 → TFT 切聊天页 → 进入 AWAKE
LLM 40s 无回复 → 超时回到 IDLE，需重新唤醒
LLM 交互中听到唤醒词 → 强制中断 LLM → 播放语音应答 → AWAKE
"""

import base64
import os
import tempfile
import threading
import time
import wave
import yaml
from collections import deque
from enum import Enum, auto

from openai import OpenAI
from services.tool_dispatcher import get_tools, ToolCallAccumulator, build_action_cmd
from .audio_pipeline import AudioPipeline
from .multimodal import create_multimodal


class _State(Enum):
    IDLE = auto()
    AWAKE = auto()
    LLM_PENDING = auto()


class VoiceChatService:
    """直接语音对话服务（唤醒词版）。

    Usage:
        vc = VoiceChatService(config_path="core/config.yaml")
        vc.on_wake_word    = your_wake_handler
        vc.on_llm_reply    = lambda text: your_tts(text)
        vc.on_llm_chunk    = lambda text: stream_tts(text)
        vc.on_tool_call    = lambda name, args: your_action(name, args)
        vc.on_llm_timeout  = your_timeout_handler
        vc.start()
    """

    SAMPLE_RATE = AudioPipeline.SAMPLE_RATE
    API_TIMEOUT = 10.0
    LLM_IDLE_TIMEOUT = 40.0

    def __init__(self, config_path="core/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_cfg = config["llm"]
        self.client = OpenAI(api_key=llm_cfg["key"], base_url=llm_cfg["url"])
        self.multimodal = create_multimodal(config_path)
        self.model = llm_cfg["model"]
        self.max_tokens = llm_cfg.get("max_tokens", 1024)
        self.system_prompt = config.get("system_prompt", "")

        # 对话历史（最近20轮）
        self._chat_history: deque = deque(maxlen=40)

        # 唤醒应答 WAV
        ww_cfg = config.get("wake_word", {})
        self._wake_response_wav = ww_cfg.get("response_wav", "assets/wake_response.wav")

        # ── AudioPipeline：统一处理采集+唤醒词+VAD断句 ──
        self._pipe = AudioPipeline(config_path)
        self._pipe.on_wake_word = self._on_wake_detected
        self._pipe.on_sentence = self._on_sentence

        # ── 状态机 ──
        self._state = _State.IDLE
        self._state_lock = threading.Lock()
        self._last_llm_activity = 0.0
        self._cancel_llm = threading.Event()
        self._llm_thread = None

        # ── 回调 ──
        self.on_wake_word = None       # 唤醒词触发（应播放应答语音、切 TFT 页面）
        self.on_llm_reply = None       # LLM 文本回复（最终完整回复）
        self.on_llm_chunk = None       # LLM 流式文本块
        self.on_tool_call = None       # LLM 工具调用
        self.on_llm_timeout = None     # 40s 无回复超时

    # ================================================================
    # Public API
    # ================================================================
    def start(self):
        self._pipe.start()
        self._last_llm_activity = time.time()

        # 超时监控线程
        threading.Thread(target=self._timeout_watch, daemon=True).start()

        print(f"[VoiceChat] 已启动 (直接语音对话)")

    def stop(self):
        self._cancel_llm.set()
        self._pipe.stop()
        if self._llm_thread and self._llm_thread.is_alive():
            self._llm_thread.join(timeout=3.0)
        print("[VoiceChat] 已停止")

    def pause(self):
        self._pipe.pause()
        self._cancel_llm.set()
        with self._state_lock:
            self._state = _State.IDLE
        print("[VoiceChat] 已暂停")

    def resume(self):
        self._pipe.resume()
        print("[VoiceChat] 已恢复")

    # ================================================================
    # 状态机入口
    # ================================================================
    def _on_wake_detected(self):
        """唤醒词触发：AudioPipeline 回调（已在音频线程内）。"""
        now = time.time()

        # 如果 LLM 正在跑，中断它
        if self._llm_thread and self._llm_thread.is_alive():
            print("[VoiceChat] 强制中断当前 LLM 对话")
            self._cancel_llm.set()
            self._llm_thread.join(timeout=2.0)

        # 进入 AWAKE
        with self._state_lock:
            self._state = _State.AWAKE
        self._last_llm_activity = now

        print(f"[VoiceChat] 唤醒成功, 距上次 {time.time() - self._last_llm_activity:.1f}s")

        # 通知外部
        if self.on_wake_word:
            try:
                self.on_wake_word()
            except Exception as e:
                print(f"[VoiceChat] on_wake_word 异常: {e}")

    def _on_sentence(self, pcm_data: bytes):
        """VAD 断句回调：仅 AWAKE 状态时派发 LLM。"""
        with self._state_lock:
            state = self._state

        if state != _State.AWAKE:
            return  # IDLE 或 LLM_PENDING 时忽略

        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE
        if duration_ms < 200:
            return

        # 转 WAV → base64，在新线程发 LLM
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="vc_")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            # 调试副本
            import shutil
            debug_dir = os.path.expanduser("~/.wali_debug")
            os.makedirs(debug_dir, exist_ok=True)
            shutil.copy2(wav_path, os.path.join(debug_dir, "vc_debug_last.wav"))

            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            self._dispatch_llm(audio_b64)

        except Exception as e:
            print(f"[VoiceChat] 语音编码失败: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _on_timeout(self):
        """LLM 超时，回到 IDLE。"""
        print("[VoiceChat] LLM 40s 无回复，超时回到待机")
        self._cancel_llm.set()
        with self._state_lock:
            self._state = _State.IDLE
        if self.on_llm_timeout:
            try:
                self.on_llm_timeout()
            except Exception as e:
                print(f"[VoiceChat] on_llm_timeout 异常: {e}")

    def _timeout_watch(self):
        """后台线程：定期检查 LLM 超时。"""
        while True:
            time.sleep(2)
            with self._state_lock:
                state = self._state
            if state in (_State.AWAKE, _State.LLM_PENDING):
                if time.time() - self._last_llm_activity > self.LLM_IDLE_TIMEOUT:
                    self._on_timeout()

    # ================================================================
    # LLM 调度
    # ================================================================
    def _dispatch_llm(self, audio_b64: str):
        """将已编码音频派发给后台 LLM 线程。"""
        self._cancel_llm.clear()
        self._llm_thread = threading.Thread(
            target=self._send_to_llm, args=(audio_b64,), daemon=True
        )
        with self._state_lock:
            self._state = _State.LLM_PENDING
        self._llm_thread.start()

    def _send_to_llm(self, audio_b64: str):
        """后台线程：拼 messages → 调 LLM → 流式回调。"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(list(self._chat_history))
        messages.append(self.multimodal.build_audio_message(audio_b64))

        print(f"[VoiceChat] 发送音频 → {self.model}")
        t0 = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text"],
                tools=get_tools(),
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
                timeout=self.API_TIMEOUT,
                max_tokens=self.max_tokens,
                frequency_penalty=0.3,
                presence_penalty=0.3,
            )

            acc = ToolCallAccumulator()
            chunks = []

            for chunk in response:
                if self._cancel_llm.is_set():
                    print("[VoiceChat] LLM 调用被中断")
                    if hasattr(response, "close"):
                        try:
                            response.close()
                        except Exception:
                            pass
                    self._llm_done()
                    return

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    acc.feed(delta)
                    if hasattr(delta, "content") and delta.content:
                        text = delta.content
                        chunks.append(text)
                        if self.on_llm_chunk:
                            self.on_llm_chunk(text)

            elapsed = time.time() - t0
            reply = "".join(chunks).strip()
            print(f"[VoiceChat] LLM 回复 ({elapsed:.1f}s): {reply}")

            # 解析 you/asr 文本存入对话历史
            asr_text = ""
            ai_text = reply
            if reply.startswith("you:"):
                lines = reply.split("\n", 1)
                asr_text = lines[0][4:].strip()
                ai_text = lines[1].strip() if len(lines) > 1 else ""
                if ai_text.startswith("ai:"):
                    ai_text = ai_text[3:].strip()
            if asr_text:
                self._chat_history.append({"role": "user", "content": asr_text})
            if ai_text:
                self._chat_history.append({"role": "assistant", "content": ai_text})

            for tc in acc.flush():
                print(f"[VoiceChat] 工具调用: {tc['name']}({tc['arguments']})")
                if self.on_tool_call:
                    self.on_tool_call(tc["name"], tc["arguments"])

            # 通知外部完整回复
            if self.on_llm_reply:
                self.on_llm_reply(reply)

        except Exception as e:
            print(f"[VoiceChat] LLM 调用失败: {e}")
        finally:
            self._llm_done()

    def _llm_done(self):
        """LLM 调用结束，回到 AWAKE。"""
        self._last_llm_activity = time.time()
        with self._state_lock:
            if self._state == _State.LLM_PENDING:
                self._state = _State.AWAKE
