"""直接语音对话服务：唤醒词守门 → VAD → 断句 → 多模态 LLM（音频直入）→ TTS

2025-06 集成 Sherpa-ONNX 唤醒词检测：
  未唤醒态 → 音频只喂给 KeywordSpotter，检测到"瓦力瓦力"后切换为唤醒态
  唤醒态   → 走原有 VAD + 断句 + Qwen-Omni 多模态链路
  唤醒后 N 秒无语音 → 自动回到未唤醒态（_awake_timeout）
"""

import base64
import os
import queue
import tempfile
import threading
import time
import wave
import yaml

import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI
from services.tool_dispatcher import get_tools, ToolCallAccumulator, build_action_cmd


class VoiceChatService:
    """直接语音对话服务。

    Usage:
        vc = VoiceChatService(config_path="core/config.yaml")
        vc.on_llm_reply = lambda text: your_tts(text)
        vc.start()
        ...
        vc.stop()
        vc.pause()   # 机器人说话时暂停，防回声
        vc.resume()
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)          # 480 samples
    FRAME_BYTES = FRAME_SIZE * 2                              # 960 bytes
    SILENCE_SEC = 0.8
    MAX_SPEECH_SEC = 15.0
    API_TIMEOUT = 10.0

    def __init__(self, config_path="core/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ai = config["ai_settings"]
        self.client = OpenAI(api_key=ai["api_key"], base_url=ai["base_url"])
        self.model = ai["model"]
        self.system_prompt = config.get("system_prompt", "")

        # ── 唤醒词配置 ──
        ww_cfg = config.get("wake_word", {})
        self._wake_word_enabled = ww_cfg.get("enabled", False)
        self._ww_keyword = ww_cfg.get("keyword", "瓦力瓦力")
        self._ww_model_dir = ww_cfg.get("model_dir", "models/sherpa-onnx")
        self._ww_threshold = ww_cfg.get("threshold", 0.5)
        self._awake_timeout = ww_cfg.get("awake_timeout", 8.0)

        self._kw_spotter = None
        self._kw_stream = None
        self._awake = False
        self._awake_since = 0.0

        if self._wake_word_enabled:
            self._init_wake_word()

        self.vad = webrtcvad.Vad(2)
        self.audio_queue = queue.Queue(maxsize=300)

        self.is_running = False
        self.is_paused = False
        self._listen_thread = None
        self._audio_stream = None
        self._paused_event = threading.Event()

        # 回调：LLM 返回文本后由调用方决定怎么处理（通常喂给 TTS）
        self.on_llm_reply = None

        # 回调：LLM 返回工具调用后由调用方决定怎么处理（通常发到 /action_cmd）
        self.on_tool_call = None

    # ================================================================
    # 唤醒词初始化
    # ================================================================
    def _init_wake_word(self):
        """加载 Sherpa-ONNX KeywordSpotter 模型。"""
        try:
            import sherpa_onnx
        except ImportError:
            print("[VoiceChat] sherpa-onnx 未安装，唤醒词功能已禁用")
            print("[VoiceChat] 安装: pip install sherpa-onnx")
            self._wake_word_enabled = False
            return

        tokens = os.path.join(self._ww_model_dir, "tokens.txt")
        encoder = os.path.join(self._ww_model_dir, "encoder-epoch-99-avg-1.onnx")
        decoder = os.path.join(self._ww_model_dir, "decoder-epoch-99-avg-1.onnx")
        joiner = os.path.join(self._ww_model_dir, "joiner-epoch-99-avg-1.onnx")
        keywords_file = os.path.join(self._ww_model_dir, "keywords.txt")

        # 自动生成 keywords.txt
        if not os.path.exists(keywords_file):
            os.makedirs(self._ww_model_dir, exist_ok=True)
            pinyin = self._ww_keyword.replace("瓦力瓦力", "wa li wa li")
            if pinyin == self._ww_keyword:
                pinyin = "wa li wa li"
            with open(keywords_file, "w", encoding="utf-8") as f:
                f.write(f"{pinyin} @{self._ww_keyword}\n")
            print(f"[VoiceChat] 已生成 keywords.txt: {keywords_file}")

        missing = [f for f in [tokens, encoder, decoder, joiner] if not os.path.exists(f)]
        if missing:
            print(f"[VoiceChat] 唤醒词模型文件缺失: {missing}")
            print(f"[VoiceChat] 模型目录: {self._ww_model_dir}")
            print("[VoiceChat] 下载: pip install sherpa-onnx 后运行 "
                  "download_sherpa_kws_model.py")
            self._wake_word_enabled = False
            return

        try:
            self._kw_spotter = sherpa_onnx.KeywordSpotter(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                keywords_file=keywords_file,
                keywords_threshold=self._ww_threshold,
                num_threads=1,
            )
            self._kw_stream = self._kw_spotter.create_stream()
            print(f"[VoiceChat] 唤醒词已就绪: '{self._ww_keyword}' "
                  f"(threshold={self._ww_threshold})")
        except Exception as e:
            print(f"[VoiceChat] 初始化唤醒词失败: {e}")
            self._wake_word_enabled = False

    # ================================================================
    # Public API（与 stt_service 完全兼容）
    # ================================================================
    def start(self):
        self.is_running = True
        self._paused_event.clear()
        self._listen_thread = threading.Thread(target=self._run, daemon=True)
        self._listen_thread.start()
        self._audio_stream = sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.SAMPLE_RATE,
            blocksize=self.FRAME_SIZE,
            callback=self._audio_callback,
        )
        self._audio_stream.start()
        tag = "唤醒词 + Qwen-Omni" if self._wake_word_enabled else "Qwen-Omni"
        print(f"[VoiceChat] 直接语音对话已启动 ({tag})")
        return True

    def stop(self):
        self.is_running = False
        self._paused_event.set()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._drain_queue()
        print("[VoiceChat] 已停止")

    def pause(self):
        self.is_paused = True
        self._paused_event.set()
        self._drain_queue()
        print("[VoiceChat] 麦克风已暂停（防回声）")

    def resume(self):
        self._drain_queue()
        self.is_paused = False
        self._paused_event.clear()
        print("[VoiceChat] 麦克风已恢复")

    # ================================================================
    # 音频采集（与 stt_service 完全一致）
    # ================================================================
    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        # 唤醒态重置
        if self._wake_word_enabled:
            self._reset_wake_state()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self.is_running or self.is_paused:
            return
        try:
            int16 = (indata[:, 0] * 32767).astype(np.int16)
            self.audio_queue.put_nowait(int16.tobytes())
        except queue.Full:
            pass

    def _vad_check(self, frame):
        try:
            return self.vad.is_speech(frame, self.SAMPLE_RATE)
        except Exception:
            return False

    # ================================================================
    # 唤醒词状态管理
    # ================================================================
    def _reset_wake_state(self):
        """回到「未唤醒」状态，重建 sherpa-onnx stream。"""
        self._awake = False
        self._awake_since = 0.0
        if self._kw_spotter and self._kw_stream:
            try:
                self._kw_spotter.reset(self._kw_stream)
            except Exception:
                pass

    def _check_wake_word(self, frame):
        """将单帧 int16 PCM 喂入 Sherpa-ONNX，返回是否检测到唤醒词。"""
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._kw_stream.accept_waveform(self.SAMPLE_RATE, samples)
        except Exception:
            return False

        detected = False
        try:
            while self._kw_spotter.is_ready(self._kw_stream):
                self._kw_spotter.decode(self._kw_stream)
                result = self._kw_spotter.get_result(self._kw_stream)
                if result and hasattr(result, 'keyword') and result.keyword:
                    print(f"[VoiceChat] 🎤 唤醒成功: {result.keyword}")
                    detected = True
                    break
        except Exception:
            pass

        return detected

    # ================================================================
    # 主循环：唤醒词守门 → VAD + 断句 → LLM
    # ================================================================
    def _run(self):
        max_silence = int(self.SILENCE_SEC / (self.FRAME_MS / 1000.0))
        max_frames = int(self.MAX_SPEECH_SEC / (self.FRAME_MS / 1000.0))

        byte_buf = bytearray()
        speech_frames = []
        silence_count = 0
        in_speech = False
        speech_frame_count = 0

        while self.is_running:
            if self._paused_event.is_set():
                time.sleep(0.1)
                byte_buf.clear()
                speech_frames.clear()
                in_speech = False
                silence_count = 0
                speech_frame_count = 0
                continue

            try:
                byte_buf.extend(self.audio_queue.get(timeout=0.1))
            except queue.Empty:
                # 唤醒超时检查
                if self._wake_word_enabled and self._awake:
                    if time.time() - self._awake_since > self._awake_timeout:
                        print("[VoiceChat] 唤醒超时，回到待机")
                        self._reset_wake_state()
                pass

            while len(byte_buf) >= self.FRAME_BYTES:
                frame = bytes(byte_buf[:self.FRAME_BYTES])
                del byte_buf[:self.FRAME_BYTES]

                # ── 唤醒词守门 ──
                if self._wake_word_enabled and not self._awake:
                    if self._check_wake_word(frame):
                        self._awake = True
                        self._awake_since = time.time()
                        # 唤醒后清空 VAD 缓冲区，只保留当前帧之后的数据
                        speech_frames.clear()
                        in_speech = False
                        silence_count = 0
                        speech_frame_count = 0
                    continue  # 未唤醒时不做 VAD

                # ── VAD + 断句（唤醒态或未启用唤醒词） ──
                is_speech = self._vad_check(frame)

                if is_speech:
                    silence_count = 0
                    if not in_speech:
                        in_speech = True
                        speech_frames.clear()
                        speech_frame_count = 0
                        print("[VoiceChat] 检测到人声，开始录音")
                    speech_frames.append(frame)
                    speech_frame_count += 1

                    if speech_frame_count >= max_frames:
                        print("[VoiceChat] 达到最大录音时长，强制断句")
                        self._send_to_llm(speech_frames)
                        speech_frames.clear()
                        in_speech = False
                        speech_frame_count = 0

                elif in_speech:
                    silence_count += 1
                    speech_frames.append(frame)
                    speech_frame_count += 1

                    if silence_count > max_silence or speech_frame_count >= max_frames:
                        in_speech = False
                        silence_count = 0
                        trim_count = min(speech_frame_count, max_silence)
                        trimmed = (
                            speech_frames[:-trim_count]
                            if trim_count < len(speech_frames)
                            else speech_frames
                        )
                        if trimmed:
                            self._send_to_llm(trimmed)
                        speech_frames.clear()
                        speech_frame_count = 0

    # ================================================================
    # 云端多模态 LLM 调用（唯一与 stt_service 不同的部分）
    # ================================================================
    def _send_to_llm(self, frames):
        """PCM 帧拼接 → 写 WAV → base64 → POST 多模态 LLM → 拿到回复文本"""

        pcm_data = b"".join(frames)
        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE
        if duration_ms < 200:
            return

        wav_path = None
        try:
            # 写临时 WAV
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="vc_")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            # 调试备份
            import shutil
            debug_dir = os.path.expanduser("~/.wali_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "vc_debug_last.wav")
            shutil.copy2(wav_path, debug_path)
            print(f"[VoiceChat] 调试音频已保存 {debug_path} ({duration_ms}ms)")

            # 读 WAV → base64
            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            # 构造多模态消息（Omni 要求 input_audio 嵌套 data/format）
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请回复这段语音。"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{audio_b64}",
                                "format": "wav",
                            },
                        },
                    ],
                },
            ]

            print(f"[VoiceChat] 发送音频 {duration_ms}ms 到 {self.model} ...")
            t0 = time.time()

            # Omni 强制 stream=True
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                modalities=["text"],
                tools=get_tools(),
                tool_choice="auto",
                stream=True,
                stream_options={"include_usage": True},
                timeout=self.API_TIMEOUT,
            )

            elapsed = time.time() - t0
            acc = ToolCallAccumulator()
            chunks = []
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    acc.feed(delta)
                    if hasattr(delta, "content") and delta.content:
                        chunks.append(delta.content)
            reply = "".join(chunks).strip()

            print(f"[VoiceChat] LLM 回复 ({elapsed:.1f}s): {reply}")

            if reply and self.on_llm_reply:
                self.on_llm_reply(reply)

            # 处理工具调用
            for tc in acc.flush():
                print(f"[VoiceChat] 工具调用: {tc['name']}({tc['arguments']})")
                if self.on_tool_call:
                    self.on_tool_call(tc["name"], tc["arguments"])

        except Exception as e:
            print(f"[VoiceChat] LLM 调用失败: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            # LLM 回复完成后，回到未唤醒态，等待下一次唤醒
            if self._wake_word_enabled:
                self._reset_wake_state()
