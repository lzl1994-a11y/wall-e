"""直接语音对话服务：唤醒词守门 → 语音应答 → VAD → 多模态 LLM → TTS

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
import queue
import tempfile
import threading
import time
import wave
import yaml
from enum import Enum, auto

import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI
from services.tool_dispatcher import get_tools, ToolCallAccumulator, build_action_cmd


class _State(Enum):
    IDLE = auto()
    AWAKE = auto()
    LLM_PENDING = auto()


class VoiceChatService:
    """直接语音对话服务（唤醒词版）。

    新增回调:
        vc.on_wake_word    = lambda: play_wav()     # 唤醒词检测到
        vc.on_llm_timeout  = lambda: switch_idle()   # 40s 无回复超时

    Usage:
        vc = VoiceChatService(config_path="core/config.yaml")
        vc.on_wake_word    = your_wake_handler
        vc.on_llm_reply    = lambda text: your_tts(text)
        vc.on_tool_call    = lambda name, args: your_action(name, args)
        vc.on_llm_timeout  = your_timeout_handler
        vc.start()
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2
    SILENCE_SEC = 0.8
    MAX_SPEECH_SEC = 15.0
    API_TIMEOUT = 10.0
    LLM_IDLE_TIMEOUT = 40.0  # LLM 无回复超时

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

        # 唤醒应答 WAV（预合成）
        self._wake_response_wav = ww_cfg.get("response_wav", "assets/wake_response.wav")

        self._kw_spotter = None
        self._kw_stream = None
        self._wake_cooldown_until = 0.0  # 唤醒冷却期，避免连续误触发

        if self._wake_word_enabled:
            self._init_wake_word()

        self.vad = webrtcvad.Vad(2)          # 断句用，中等灵敏度
        self._ww_vad = webrtcvad.Vad(0)      # 唤醒词前置滤网，最宽松模式
        self.audio_queue = queue.Queue(maxsize=300)

        self.is_running = False
        self.is_paused = False
        self._listen_thread = None
        self._audio_stream = None
        self._paused_event = threading.Event()

        # ── 状态机 ──
        self._state = _State.IDLE
        self._state_lock = threading.Lock()
        self._awake_since = 0.0
        self._last_llm_activity = 0.0
        self._cancel_llm = threading.Event()
        self._llm_thread = None

        # ── 回调 ──
        self.on_wake_word = None       # 唤醒词触发（应播放应答语音、切 TFT 页面）
        self.on_llm_reply = None       # LLM 文本回复
        self.on_tool_call = None       # LLM 工具调用
        self.on_llm_timeout = None     # 40s 无回复超时

        # ── 调试 ──
        self._debug_ring = []  # 最近 N 帧音频 (int16 bytes)，用于调试保存
        self._debug_ring_max = int(5.0 / (self.FRAME_MS / 1000.0))  # 5 秒
        self._heartbeat_at = 0.0

    # ================================================================
    # 唤醒词初始化
    # ================================================================
    def _init_wake_word(self):
        try:
            import sherpa_onnx
        except ImportError:
            print("[VoiceChat] sherpa-onnx 未安装，唤醒词功能已禁用")
            print("[VoiceChat] 安装: pip install sherpa-onnx")
            self._wake_word_enabled = False
            return

        import glob as _glob

        tokens = os.path.join(self._ww_model_dir, "tokens.txt")
        keywords_file = os.path.join(self._ww_model_dir, "keywords.txt")

        # 用 glob 匹配实际文件名（不硬编码版本号），优先非 int8
        def _pick_nonint8(pattern):
            files = _glob.glob(pattern)
            nonint8 = [f for f in files if "int8" not in os.path.basename(f)]
            return nonint8 if nonint8 else files

        _enc = _pick_nonint8(os.path.join(self._ww_model_dir, "encoder-*.onnx"))
        _dec = _pick_nonint8(os.path.join(self._ww_model_dir, "decoder-*.onnx"))
        _joi = _pick_nonint8(os.path.join(self._ww_model_dir, "joiner-*.onnx"))

        if not (_enc and _dec and _joi and os.path.exists(tokens)):
            missing = []
            if not _enc: missing.append("encoder-*.onnx")
            if not _dec: missing.append("decoder-*.onnx")
            if not _joi: missing.append("joiner-*.onnx")
            if not os.path.exists(tokens): missing.append("tokens.txt")
            print(f"[VoiceChat] 唤醒词模型缺失: {missing}")
            print(f"[VoiceChat] 请运行 download_sherpa_kws_model.py 下载模型")
            self._wake_word_enabled = False
            return

        encoder = _enc[0]
        decoder = _dec[0]
        joiner = _joi[0]

        if not os.path.exists(keywords_file):
            os.makedirs(self._ww_model_dir, exist_ok=True)
            with open(keywords_file, "w", encoding="utf-8") as f:
                f.write("w a l i w a l i @瓦力瓦力\n"  # BPE 子词分词，模型 tokens.txt 里没有 wa/li 完整拼音
                        "w a n i w a n i @瓦力瓦力\n"
                        "w a l íng w a l íng @瓦力瓦力\n"
                        "w a y i w a y i @瓦力瓦力\n"
                        "w a l èi w a l èi @瓦力瓦力\n")

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
            print(f"[VoiceChat] 唤醒词就绪: '{self._ww_keyword}' "
                  f"(threshold={self._ww_threshold})")
        except Exception as e:
            print(f"[VoiceChat] 初始化唤醒词失败: {e}")
            self._wake_word_enabled = False

    # ================================================================
    # Public API
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
        print(f"[VoiceChat] 已启动 ({tag})")

        # 打印音频设备信息
        try:
            dev = sd.query_devices(kind="input")
            print(f"[VoiceChat] 输入设备: {dev['name']} | "
                  f"默认采样率:{int(dev['default_samplerate'])} | "
                  f"通道数:{dev['max_input_channels']}")
        except Exception:
            print("[VoiceChat] 无法查询输入设备信息")

        return True

    def stop(self):
        self.is_running = False
        self._cancel_llm.set()
        self._paused_event.set()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        if self._llm_thread and self._llm_thread.is_alive():
            self._llm_thread.join(timeout=3.0)
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._drain_queue()
        print("[VoiceChat] 已停止")

    def pause(self):
        self.is_paused = True
        self._paused_event.set()
        self._cancel_llm.set()
        self._drain_queue()
        with self._state_lock:
            self._state = _State.IDLE
        print("[VoiceChat] 已暂停")

    def resume(self):
        self._drain_queue()
        self.is_paused = False
        self._paused_event.clear()
        print("[VoiceChat] 已恢复")

    # ================================================================
    # 音频采集
    # ================================================================
    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

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
    # 唤醒词检测 + 状态管理
    # ================================================================
    def _check_wake_word(self, frame):
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._kw_stream.accept_waveform(self.SAMPLE_RATE, samples)
        except Exception:
            return False
        try:
            while self._kw_spotter.is_ready(self._kw_stream):
                print("[DEBUG_KWS] spotter methods:", [m for m in dir(self._kw_spotter) if not m.startswith('_')])
                self._kw_spotter.decode(self._kw_stream)
                result = self._kw_spotter.get_result(self._kw_stream)
                # ── 临时调试：打印每次就绪的搜索结果 ──
                print(f"[DEBUG_KWS] is_ready=True, result={result}, keyword={result.keyword if (result and hasattr(result,'keyword')) else 'N/A'}")
                if result and hasattr(result, 'keyword') and result.keyword:
                    return True
        except Exception as e:
            print(f"[DEBUG_KWS] 异常: {e}")
        return False

    def _on_wake_detected(self):
        """唤醒词触发后的处理。"""
        # 冷却检查
        now = time.time()
        if now < self._wake_cooldown_until:
            return
        self._wake_cooldown_until = now + 1.5  # 1.5s 冷却

        print(f"[VoiceChat] 🎤 唤醒成功: {self._ww_keyword}  "
              f"(冷却剩余 {max(0, self._wake_cooldown_until - now):.1f}s, "
              f"距上次唤醒 {now - self._awake_since:.1f}s)")

        # 如果有正在进行的 LLM 调用，强制中断
        if self._llm_thread and self._llm_thread.is_alive():
            print("[VoiceChat] 强制中断当前 LLM 对话")
            self._cancel_llm.set()
            self._llm_thread.join(timeout=2.0)

        # 进入 AWAKE 状态
        with self._state_lock:
            self._state = _State.AWAKE
        self._awake_since = now
        self._last_llm_activity = now

        # 重置唤醒词流，防止持续误触发
        if self._kw_spotter:
            try:
                self._kw_stream = self._kw_spotter.create_stream()
            except Exception:
                pass

        # 保存调试音频（最近 5 秒）
        self._save_debug_audio("wake_trigger")

        # 通知外部：播放应答语音 + 切 TFT
        if self.on_wake_word:
            try:
                self.on_wake_word()
            except Exception as e:
                print(f"[VoiceChat] on_wake_word 回调异常: {e}")

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
                print(f"[VoiceChat] on_llm_timeout 回调异常: {e}")

    def _save_debug_audio(self, tag="debug"):
        """保存调试环缓冲中最近几秒的音频到 ~/.wali_debug/ 目录。"""
        if not self._debug_ring:
            return
        try:
            import wave as _wave
            debug_dir = os.path.expanduser("~/.wali_debug")
            os.makedirs(debug_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(debug_dir, f"ww_{tag}_{ts}.wav")
            pcm = b"".join(self._debug_ring)
            with _wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm)
            dur = len(pcm) / (self.SAMPLE_RATE * 2)
            print(f"[VoiceChat] 调试音频已保存: {path} ({dur:.1f}s)")
        except Exception as e:
            print(f"[VoiceChat] 保存调试音频失败: {e}")

    # ================================================================
    # 主循环
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
                pass

            # ── 调试心跳：每 2 秒打印，确认音频流在工作 ──
            now_hb = time.time()
            if now_hb - self._heartbeat_at > 2.0:
                self._heartbeat_at = now_hb
                qs = self.audio_queue.qsize()
                state_name = self._state.name
                # 计算最近 1 秒的电平 (RMS)
                rms_db = -999
                if self._debug_ring:
                    recent = self._debug_ring[-min(50, len(self._debug_ring)):]
                    pcm = b"".join(recent)
                    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(arr ** 2))
                    rms_db = 20 * np.log10(max(rms, 1e-6))
                print(f"[VoiceChat] ♥ 音频流活跃 | 队列:{qs} | 状态:{state_name} | "
                      f"电平:{rms_db:.0f}dB | 环缓冲:{len(self._debug_ring)}帧")

            # 超时检查（AWAKE / LLM_PENDING 状态）
            with self._state_lock:
                current_state = self._state
            if current_state in (_State.AWAKE, _State.LLM_PENDING):
                if time.time() - self._last_llm_activity > self.LLM_IDLE_TIMEOUT:
                    self._on_timeout()

            while len(byte_buf) >= self.FRAME_BYTES:
                frame = bytes(byte_buf[:self.FRAME_BYTES])
                del byte_buf[:self.FRAME_BYTES]

                # 调试环缓冲
                self._debug_ring.append(frame)
                if len(self._debug_ring) > self._debug_ring_max:
                    self._debug_ring.pop(0)

                # ── 唤醒词检测（始终运行，所有状态） ──
                # 前置 VAD 滤网：静音帧不喂入模型，省算力
                if self._wake_word_enabled and self._ww_vad.is_speech(frame, self.SAMPLE_RATE):
                    if self._check_wake_word(frame):
                        self._on_wake_detected()
                        # 清空 VAD 缓冲，从头开始听
                        speech_frames.clear()
                        in_speech = False
                        silence_count = 0
                        speech_frame_count = 0
                        continue

                # ── 非 AWAKE 状态跳过 VAD ──
                with self._state_lock:
                    current_state = self._state
                if current_state != _State.AWAKE:
                    continue

                # ── VAD + 断句 ──
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
                        self._dispatch_llm(speech_frames)
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
                            self._dispatch_llm(trimmed)
                        speech_frames.clear()
                        speech_frame_count = 0

    # ================================================================
    # LLM 调度（解耦为独立线程）
    # ================================================================
    def _dispatch_llm(self, frames):
        """将语音帧派发给后台 LLM 线程，主循环继续处理音频。"""
        self._cancel_llm.clear()
        self._llm_thread = threading.Thread(
            target=self._send_to_llm, args=(frames,), daemon=True
        )
        with self._state_lock:
            self._state = _State.LLM_PENDING
        self._llm_thread.start()

    def _send_to_llm(self, frames):
        """后台线程：PCM → WAV → base64 → Qwen-Omni → 回调。"""
        pcm_data = b"".join(frames)
        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE
        if duration_ms < 200:
            self._llm_done()
            return

        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="vc_")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            import shutil
            debug_dir = os.path.expanduser("~/.wali_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "vc_debug_last.wav")
            shutil.copy2(wav_path, debug_path)
            print(f"[VoiceChat] 调试音频: {debug_path} ({duration_ms}ms)")

            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

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

            print(f"[VoiceChat] 发送音频 {duration_ms}ms → {self.model}")
            t0 = time.time()

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

            acc = ToolCallAccumulator()
            chunks = []
            for chunk in response:
                if self._cancel_llm.is_set():
                    print("[VoiceChat] LLM 调用被唤醒词中断")
                    # 尝试关闭底层连接
                    if hasattr(response, 'close'):
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
                        chunks.append(delta.content)

            elapsed = time.time() - t0
            reply = "".join(chunks).strip()
            print(f"[VoiceChat] LLM 回复 ({elapsed:.1f}s): {reply}")

            if reply and self.on_llm_reply:
                self.on_llm_reply(reply)

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
            self._llm_done()

    def _llm_done(self):
        """LLM 调用结束，回到 AWAKE 状态。"""
        self._last_llm_activity = time.time()
        with self._state_lock:
            if self._state == _State.LLM_PENDING:
                self._state = _State.AWAKE
