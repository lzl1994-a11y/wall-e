"""
共享音频管线：麦克风采集 → 唤醒词守门(可选) → VAD 断句 → 回调

Usage:
    from services.audio_pipeline import AudioPipeline
    pipe = AudioPipeline(config_path)
    pipe.on_speech_start = lambda initial_pcm: ...
    pipe.on_speech_audio = lambda pcm_frame: ...
    pipe.on_sentence = lambda pcm_frames: ...
    pipe.on_wake_word = lambda: ...
    pipe.start()
"""

import os
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import yaml

from services.usb_devices import resolve_audio_device


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
    on_speech_start: Callable[[bytes], None] — 开口时回调，包含预录音和首帧
    on_speech_audio: Callable[[bytes], None] — 开口后的连续 PCM 帧
    on_speech_cancel: Callable[[], None] — 过短语音或中途取消
    on_wake_word: Callable[[], None]     — 唤醒词触发（仅 enabled=True 时）
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30  # WebRTC VAD 严格要求 10ms, 20ms, 或 30ms
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2
    PRE_ROLL_SEC = 0.3
    SILENCE_SEC = 0.5
    MAX_SPEECH_SEC = 15.0

    def __init__(self, config_path: str = "core/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._ww = WakeWordDetector(config)
        self._config_path = config_path
        
        self._vad_cfg = config.get("vad", {})
        if not isinstance(self._vad_cfg, dict):
            self._vad_cfg = {}
        self._vad_backend = "none"
        self._vad = None
        self._has_webrtc = False
        self._silero_session = None
        self._silero_state = None
        self._init_vad()
            
        self._vad_lock = threading.Lock()
        self._vad_err_count = 0
        # 断句 VAD 阈值
        self._vad_thresh = float(self._vad_cfg.get("threshold", 0.5))
        try:
            silence_sec = float(self._vad_cfg.get("silence_sec", self.SILENCE_SEC))
        except (TypeError, ValueError):
            silence_sec = self.SILENCE_SEC
        self._silence_sec = min(2.0, max(0.3, silence_sec))

        self.audio_queue = queue.Queue(maxsize=300)
        self._is_running = False
        self._is_paused = False
        self._paused_event = threading.Event()
        self._listen_thread = None
        self._device_thread = None
        self._audio_stream = None
        self._audio_stream_lock = threading.Lock()
        self._audio_device_identity = ""
        self._awake = False  # 唤醒后才启动 VAD 断句

        self.on_sentence = None       # Callable[[bytes], None]
        self.on_speech_start = None   # Callable[[bytes], None]
        self.on_speech_audio = None   # Callable[[bytes], None]
        self.on_speech_cancel = None  # Callable[[], None]
        self.on_wake_word = None      # Callable[[], None]

    # ── Public API ──
    def start(self):
        self._is_running = True
        self._paused_event.clear()
        self._listen_thread = threading.Thread(target=self._run, daemon=True)
        self._listen_thread.start()
        self._device_thread = threading.Thread(target=self._device_monitor, daemon=True)
        self._device_thread.start()
        print(f"[AudioPipeline] started (wake-word={'ON' if self._ww.enabled else 'OFF'})")

    def stop(self):
        self._is_running = False
        self._paused_event.set()
        self._close_audio_stream()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        if self._device_thread and self._device_thread.is_alive():
            self._device_thread.join(timeout=2.0)
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
        self._reset_vad_state()
        print("[AudioPipeline] 已恢复")

    def _init_vad(self):
        """按配置加载 VAD；依赖缺失时回退到另一个可用后端。"""
        requested = str(self._vad_cfg.get("provider", "webrtc")).strip().lower()
        if requested in {"webrtcvad", "webrtc_vad"}:
            requested = "webrtc"
        if requested in {"silero_vad", "silero-onnx"}:
            requested = "silero"
        if requested not in {"webrtc", "silero"}:
            print(f"[AudioPipeline] 未知 VAD provider={requested!r}，回退 WebRTC")
            requested = "webrtc"

        if requested == "silero":
            if self._init_silero_vad():
                return
            print("[AudioPipeline] Silero VAD 不可用，回退 WebRTC VAD")
            if self._init_webrtc_vad():
                return
        else:
            if self._init_webrtc_vad():
                return
            print("[AudioPipeline] WebRTC VAD 不可用，回退 Silero VAD")
            if self._init_silero_vad():
                return
        self._vad_backend = "none"
        print("[AudioPipeline] 没有可用的 VAD 后端，语音断句将保持关闭")

    def _init_webrtc_vad(self):
        try:
            import webrtcvad
            aggressiveness = int(self._vad_cfg.get("aggressiveness", 3))
            aggressiveness = max(0, min(3, aggressiveness))
            self._vad = webrtcvad.Vad(aggressiveness)
            self._has_webrtc = True
            self._vad_backend = "webrtc"
            print(
                f"[AudioPipeline] 加载 WebRTC VAD 成功 "
                f"(Aggressiveness={aggressiveness})"
            )
            return True
        except (ImportError, ValueError, TypeError) as exc:
            self._has_webrtc = False
            self._vad = None
            print(f"[AudioPipeline] WebRTC VAD 不可用: {exc}")
            return False

    def _init_silero_vad(self):
        model_path = str(self._vad_cfg.get("model_path", "models/silero_vad.onnx"))
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
        try:
            import onnxruntime as ort
            if not os.path.isfile(model_path):
                raise FileNotFoundError(model_path)
            self._silero_session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._silero_state = np.zeros((2, 1, 128), dtype=np.float32)
            self._vad_backend = "silero"
            print(
                f"[AudioPipeline] 加载 Silero VAD 成功 "
                f"(threshold={self._vad_cfg.get('threshold', 0.5)}, model={model_path})"
            )
            return True
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self._silero_session = None
            self._silero_state = None
            print(f"[AudioPipeline] Silero VAD 不可用: {exc}")
            return False

    def _reset_vad_state(self):
        with self._vad_lock:
            if self._vad_backend == "silero":
                self._silero_state = np.zeros((2, 1, 128), dtype=np.float32)

    # ── Internal ──
    def _close_audio_stream(self):
        with self._audio_stream_lock:
            stream = self._audio_stream
            self._audio_stream = None
            self._audio_device_identity = ""
        if stream:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _open_audio_stream(self, device_index, identity):
        for channels in (2, 1):
            stream = None
            try:
                stream = sd.InputStream(
                    device=device_index,
                    channels=channels,
                    dtype="float32",
                    samplerate=self.SAMPLE_RATE,
                    blocksize=self.FRAME_SIZE,
                    callback=self._audio_callback,
                )
                stream.start()
                with self._audio_stream_lock:
                    self._audio_stream = stream
                    self._audio_device_identity = identity
                print(f"[AudioPipeline] microphone connected (device={device_index}, channels={channels})")
                return True
            except Exception:
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
        return False

    def _device_monitor(self):
        last_wait_message = 0.0
        while self._is_running:
            resolution = resolve_audio_device(
                "input", self._config_path, sounddevice_module=sd
            )
            with self._audio_stream_lock:
                stream = self._audio_stream
                identity = self._audio_device_identity
            active = False
            if stream:
                try:
                    active = bool(stream.active)
                except Exception:
                    pass

            if not resolution.available:
                if stream:
                    self._close_audio_stream()
                now = time.monotonic()
                if now - last_wait_message >= 10.0:
                    print("[AudioPipeline] waiting for configured voice USB")
                    last_wait_message = now
            elif not active or identity != resolution.identity:
                if stream:
                    self._close_audio_stream()
                if not self._open_audio_stream(resolution.index, resolution.identity):
                    now = time.monotonic()
                    if now - last_wait_message >= 10.0:
                        print("[AudioPipeline] microphone unavailable; retrying")
                        last_wait_message = now
            time.sleep(1.0)

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
        silence_sec = getattr(self, "_silence_sec", self.SILENCE_SEC)
        max_silence = int(silence_sec / (self.FRAME_MS / 1000.0))
        max_frames = int(self.MAX_SPEECH_SEC / (self.FRAME_MS / 1000.0))
        pre_roll_size = max(1, int(self.PRE_ROLL_SEC / (self.FRAME_MS / 1000.0)))

        byte_buf = bytearray()
        pre_roll_frames = deque(maxlen=pre_roll_size)
        speech_frames = []
        silence_count = 0
        in_speech = False
        speech_frame_count = 0

        while self._is_running:
            if self._paused_event.is_set():
                time.sleep(0.1)
                byte_buf.clear()
                pre_roll_frames.clear()
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
                        self._reset_vad_state()
                        pre_roll_frames.clear()
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
                    pre_roll_frames.clear()
                    continue

                # ── VAD + 静音断句 ──
                is_speech = self._vad_prob(frame) > self._vad_thresh

                if is_speech:
                    silence_count = 0
                    speech_started = False
                    if not in_speech:
                        in_speech = True
                        speech_frames = list(pre_roll_frames)
                        speech_frame_count = len(speech_frames)
                        pre_roll_frames.clear()
                        speech_started = True
                    speech_frames.append(frame)
                    speech_frame_count += 1
                    if speech_started:
                        self._emit_speech_start(speech_frames)
                    else:
                        self._emit_speech_audio(frame)

                    if speech_frame_count >= max_frames:
                        self._emit_sentence(speech_frames)
                        speech_frames.clear()
                        in_speech = False
                        speech_frame_count = 0

                elif in_speech:
                    silence_count += 1
                    speech_frames.append(frame)
                    speech_frame_count += 1
                    self._emit_speech_audio(frame)

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
                else:
                    pre_roll_frames.append(frame)

    def _vad_prob(self, frame: bytes) -> float:
        """
        使用配置的 VAD 后端返回 0~1 的人声概率。
        """
        if self._vad_backend == "none":
            return 0.0
        try:
            if self._vad_backend == "webrtc":
                is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE)
                prob = 1.0 if is_speech else 0.0
            else:
                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                with self._vad_lock:
                    output, state = self._silero_session.run(
                        None,
                        {
                            "input": samples.reshape(1, -1),
                            "state": self._silero_state,
                            "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
                        },
                    )
                    self._silero_state = state
                prob = float(output[0][0])

            self._vad_err_count += 1
            if self._vad_err_count % 10 == 0:
                status = "人声" if prob > self._vad_thresh else "噪音/静音"
                print(f"  [{self._vad_backend.upper()} VAD] 状态: {status} ({prob:.2f})")
            return prob
        except Exception:
            return 0.0

    def _emit_speech_start(self, frames):
        callback = getattr(self, "on_speech_start", None)
        if not frames or not callback:
            return
        try:
            callback(b"".join(frames))
        except Exception as exc:
            print(f"[AudioPipeline] on_speech_start 异常: {exc}")

    def _emit_speech_audio(self, frame):
        callback = getattr(self, "on_speech_audio", None)
        if not frame or not callback:
            return
        try:
            callback(frame)
        except Exception as exc:
            print(f"[AudioPipeline] on_speech_audio 异常: {exc}")

    def _emit_speech_cancel(self):
        callback = getattr(self, "on_speech_cancel", None)
        if not callback:
            return
        try:
            callback()
        except Exception as exc:
            print(f"[AudioPipeline] on_speech_cancel 异常: {exc}")

    def _emit_sentence(self, frames):
        """将帧列表合并为 PCM bytes，触发 on_sentence 回调。"""
        if not frames:
            return
        pcm = b"".join(frames)
        dur = len(pcm) // 2 * 1000 // self.SAMPLE_RATE
        if dur < 200:
            self._emit_speech_cancel()
            return
        if not self.on_sentence:
            self._emit_speech_cancel()
            return
        try:
            self.on_sentence(pcm)
        except Exception as e:
            print(f"[AudioPipeline] on_sentence 异常: {e}")
