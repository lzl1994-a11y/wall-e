"""
共享音频管线：麦克风采集 → 唤醒词守门(可选) → VAD 断句 → 回调

Usage:
    from services.audio_pipeline import AudioPipeline
    pipe = AudioPipeline(config_path)
    pipe.on_sentence = lambda pcm_frames: ...
    pipe.on_wake_word = lambda: ...
    pipe.start()
"""

import os
import queue
import threading
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import yaml


class WakeWordDetector:
    """sherpa-onnx 唤醒词检测器。VAD 前置滤网 + 模型推理。"""

    def __init__(self, config: dict):
        ww = config.get("wake_word", {})
        if not ww.get("enabled", False):
            self._enabled = False
            return

        self._enabled = True
        self._keyword = ww.get("keyword", "瓦力瓦力")
        self._model_dir = ww.get("model_dir", "models/sherpa-onnx")
        self._threshold = ww.get("threshold", 0.5)
        self._cooldown = 1.5  # 唤醒冷却期

        import glob as _glob

        tokens = os.path.join(self._model_dir, "tokens.txt")

        def _pick(pattern):
            files = _glob.glob(pattern)
            # 过滤 int8 模型，且如果有多个模型，优先选择带有 epoch-99 的中文模型
            fp32_files = [f for f in files if "int8" not in os.path.basename(f)]
            epoch99 = [f for f in fp32_files if "epoch-99" in os.path.basename(f)]
            return epoch99 if epoch99 else (fp32_files or files)

        _enc = _pick(os.path.join(self._model_dir, "encoder-*.onnx"))
        _dec = _pick(os.path.join(self._model_dir, "decoder-*.onnx"))
        _joi = _pick(os.path.join(self._model_dir, "joiner-*.onnx"))

        if not (_enc and _dec and _joi and os.path.exists(tokens)):
            self._enabled = False
            return

        import sherpa_onnx

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=_enc[0],
            decoder=_dec[0],
            joiner=_joi[0],
            keywords_file=os.path.join(self._model_dir, "keywords.txt"),
            keywords_threshold=self._threshold,
            num_threads=1,
        )
        self._stream = self._spotter.create_stream()
        self._cooldown_until = 0.0

        print(f"[AudioPipeline] 唤醒词就绪: '{self._keyword}'")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(self, frame: bytes) -> bool:
        """喂一帧 PCM，返回是否触发唤醒词。调用方负责 VAD 前置过滤。"""
        if not self._enabled:
            return False

        now = time.time()
        if now < self._cooldown_until:
            return False

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._stream.accept_waveform(16000, samples)
        except Exception:
            return False

        try:
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
                if self._spotter.get_result(self._stream):
                    self._cooldown_until = time.time() + self._cooldown
                    self.reset()
                    return True
        except Exception:
            pass

        return False

    def reset(self):
        """重置识别流（防连续误触发）。"""
        if self._enabled and self._spotter:
            try:
                self._stream = self._spotter.create_stream()
            except Exception:
                pass


class AudioPipeline:
    """
    音频管线：采集 → 唤醒词守门(可选) → VAD 断句 → 回调。

    on_sentence: Callable[[bytes], None]  — PCM 帧列表转为连续 bytes 后回调
    on_wake_word: Callable[[], None]     — 唤醒词触发（仅 enabled=True 时）
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30  # WebRTC VAD 严格要求 10ms, 20ms, 或 30ms
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2
    SILENCE_SEC = 0.8
    MAX_SPEECH_SEC = 15.0

    def __init__(self, config_path: str = "core/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._ww = WakeWordDetector(config)
        
        # 【替换为 WebRTC VAD】：专门抵抗机械噪音 + 无视直流偏置！
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(3)  # 等级 3：最严格过滤模式，最不容易被履带噪音误触发
            self._has_webrtc = True
            print("[AudioPipeline] 加载 WebRTC VAD 成功 (Aggressiveness=3)")
        except ImportError:
            self._has_webrtc = False
            print("[AudioPipeline] 未安装 webrtcvad，请执行: pip install webrtcvad")
            
        self._vad_lock = threading.Lock()
        self._vad_err_count = 0
        # 断句 VAD 阈值
        self._vad_thresh = 0.5

        self.audio_queue = queue.Queue(maxsize=300)
        self._is_running = False
        self._is_paused = False
        self._paused_event = threading.Event()
        self._listen_thread = None
        self._audio_stream = None
        self._awake = False  # 唤醒后才启动 VAD 断句

        self.on_sentence = None     # Callable[[bytes], None]
        self.on_wake_word = None    # Callable[[], None]

    # ── Public API ──
    def start(self):
        self._is_running = True
        self._paused_event.clear()
        self._listen_thread = threading.Thread(target=self._run, daemon=True)
        self._listen_thread.start()
        
        # 【核心修复】：防止双声道麦克风（如 I2S/USB 阵列）出现“左耳静音，右耳有声”导致的漏音问题。
        # 改为尝试请求双声道并做平均混合，如果硬件不支持双声道，再退回到单声道。
        try:
            self._audio_stream = sd.InputStream(
                channels=2,
                dtype="float32",
                samplerate=self.SAMPLE_RATE,
                blocksize=self.FRAME_SIZE,
                callback=self._audio_callback,
            )
        except Exception:
            self._audio_stream = sd.InputStream(
                channels=1,
                dtype="float32",
                samplerate=self.SAMPLE_RATE,
                blocksize=self.FRAME_SIZE,
                callback=self._audio_callback,
            )
            
        self._audio_stream.start()
        print(f"[AudioPipeline] 已启动 (唤醒词={'ON' if self._ww.enabled else 'OFF'})")

    def stop(self):
        self._is_running = False
        self._paused_event.set()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._drain_queue()
        print("[AudioPipeline] 已停止")

    def set_awake(self, value: bool):
        """外部重置唤醒状态（超时后关闭 VAD）"""
        self._awake = value

    def pause(self):
        self._is_paused = True
        self._paused_event.set()
        self._drain_queue()
        print("[AudioPipeline] 已暂停")

    def resume(self):
        self._drain_queue()
        self._is_paused = False
        self._paused_event.clear()
        with self._vad_lock:
            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        print("[AudioPipeline] 已恢复")

    # ── Internal ──
    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._is_running or self._is_paused:
            return
        try:
            # 无论输入是单声道还是双声道，全部混合为单声道 (mono)
            if indata.shape[1] > 1:
                mono_audio = np.mean(indata, axis=1)
            else:
                mono_audio = indata[:, 0]
                
            # 我们需要剔除直流偏置，防止极端情况下影响 WebRTC（虽然它天然抗 DC）
            mono_audio = mono_audio - np.mean(mono_audio)
                
            int16 = (mono_audio * 32767).astype(np.int16)
            self.audio_queue.put_nowait(int16.tobytes())
        except queue.Full:
            pass

    def _run(self):
        max_silence = int(self.SILENCE_SEC / (self.FRAME_MS / 1000.0))
        max_frames = int(self.MAX_SPEECH_SEC / (self.FRAME_MS / 1000.0))

        byte_buf = bytearray()
        speech_frames = []
        silence_count = 0
        in_speech = False
        speech_frame_count = 0

        while self._is_running:
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

            while len(byte_buf) >= self.FRAME_BYTES:
                frame = bytes(byte_buf[:self.FRAME_BYTES])
                del byte_buf[:self.FRAME_BYTES]

                # ── 唤醒词检测（所有帧直送 Sherpa-ONNX，不做 VAD 前置过滤）──
                if self._ww.enabled:
                    if self._ww.check(frame):
                        print(f"[AudioPipeline] 唤醒词触发: '{self._ww._keyword}'")
                        self._awake = True
                        speech_frames.clear()
                        in_speech = False
                        silence_count = 0
                        speech_frame_count = 0
                            
                        if self.on_wake_word:
                            try:
                                self.on_wake_word()
                            except Exception as e:
                                print(f"[AudioPipeline] on_wake_word 异常: {e}")
                        continue

                # ── 未唤醒时跳过 VAD 断句 ──
                if not self._awake:
                    continue

                # ── VAD + 静音断句 ──
                is_speech = self._vad_prob(frame) > self._vad_thresh

                if is_speech:
                    silence_count = 0
                    if not in_speech:
                        in_speech = True
                        speech_frames.clear()
                        speech_frame_count = 0
                    speech_frames.append(frame)
                    speech_frame_count += 1

                    if speech_frame_count >= max_frames:
                        self._emit_sentence(speech_frames)
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
                        self._emit_sentence(trimmed)
                        speech_frames.clear()
                        speech_frame_count = 0

    def _vad_prob(self, frame: bytes) -> float:
        """
        WebRTC VAD: 基于高斯混合模型 (GMM) 的频域分析。
        能极其有效地分辨【人类语音】和【机械噪音】。
        返回 1.0 (是人声) 或 0.0 (非人声)。
        """
        if not getattr(self, '_has_webrtc', False):
            return 0.0
            
        try:
            # frame 恰好是 30ms (960 bytes) 的 16kHz PCM，完全契合 WebRTC 的胃口
            is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE)
            prob = 1.0 if is_speech else 0.0
            
            self._vad_err_count += 1
            if self._vad_err_count % 10 == 0:
                status = "🗣️ 人声" if is_speech else "🔇 噪音/静音"
                print(f"  [WebRTC VAD] 状态: {status}")
                
            return prob
        except Exception as e:
            return 0.0

    def _emit_sentence(self, frames):
        """将帧列表合并为 PCM bytes，触发 on_sentence 回调。"""
        if not frames or not self.on_sentence:
            return
        pcm = b"".join(frames)
        dur = len(pcm) // 2 * 1000 // self.SAMPLE_RATE
        if dur < 200:
            return
        try:
            self.on_sentence(pcm)
        except Exception as e:
            print(f"[AudioPipeline] on_sentence 异常: {e}")
