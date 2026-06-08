# services/stt_service.py
"""
[ZH] 阿里云 Paraformer 语音识别服务
     使用 dashscope.audio.asr.Recognition.call() 同步 HTTP 模式
     配合 WebRTC VAD 进行静音断句，断句后写入临时 WAV 文件一次性上传云端。
     避免 WebSocket 流式 API 在后台线程中的 asyncio 事件循环问题。
[EN] Alibaba Cloud Paraformer STT service (synchronous HTTP mode).
"""
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

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback


# ---------------------------------------------------------------------------
# 空回调：call() 方法必须传 callback 但不使用流式事件
# ---------------------------------------------------------------------------
class _DummyCallback(RecognitionCallback):
    pass


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------
class STTService:
    """
    外部接口保持兼容:
        stt = STTService(config_path, on_sentence_received=callback)
        stt.start()   # 开始监听
        stt.stop()    # 停止监听
        stt.pause()   # 机器人说话时暂停（防回声）
        stt.resume()  # 恢复监听
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2
    SILENCE_SEC = 0.8            # 静音断句阈值
    MAX_SPEECH_SEC = 15.0        # 单句最长时长（防止无限录制）
    API_TIMEOUT = 5.0            # 云端 API 调用超时

    def __init__(self, config_path="core/config.yaml", on_sentence_received=None):
        self.on_sentence_received = on_sentence_received
        self.is_running = False
        self.is_paused = False

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.api_key = config['ai_settings']['api_key']

        self.vad = webrtcvad.Vad(2)
        self.audio_queue = queue.Queue(maxsize=300)

        self._listen_thread = None
        self._audio_stream = None
        self._paused_event = threading.Event()

    # ===================================================================
    # Public API
    # ===================================================================
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
        print("[STT] Paraformer 语音监听已启动 (同步 HTTP 模式)")
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
        print("[STT] 语音监听已停止")

    def pause(self):
        self.is_paused = True
        self._paused_event.set()
        self._drain_queue()
        print("[STT] 麦克风已暂停 (防回声)")

    def resume(self):
        self._drain_queue()
        self.is_paused = False
        self._paused_event.clear()
        print("[STT] 麦克风已恢复")

    # ===================================================================
    # Internal
    # ===================================================================
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

    def _run(self):
        """主循环：VAD 检测 → 采集语音 → 静音断句 → 写 WAV → call() 云端识别。"""
        dashscope.api_key = self.api_key

        max_silence = int(self.SILENCE_SEC / (self.FRAME_MS / 1000.0))
        max_frames = int(self.MAX_SPEECH_SEC / (self.FRAME_MS / 1000.0))

        byte_buf = bytearray()
        speech_frames = []      # 当前句子的所有音频帧
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

            while len(byte_buf) >= self.FRAME_BYTES:
                frame = bytes(byte_buf[:self.FRAME_BYTES])
                del byte_buf[:self.FRAME_BYTES]

                is_speech = self._vad_check(frame)

                if is_speech:
                    silence_count = 0
                    if not in_speech:
                        in_speech = True
                        speech_frames.clear()
                        speech_frame_count = 0
                        print("[STT] 检测到人声，开始录制")
                    speech_frames.append(frame)
                    speech_frame_count += 1
                    # 超长保护：强制断句
                    if speech_frame_count >= max_frames:
                        print("[STT] 达到最大录制时长，强制断句")
                        self._process_speech(speech_frames)
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
                        # 剔除末尾的静音帧，减少无效数据
                        trim_count = min(speech_frame_count, max_silence)
                        trimmed = speech_frames[:-trim_count] if trim_count < len(speech_frames) else speech_frames
                        if trimmed:
                            self._process_speech(trimmed)
                        speech_frames.clear()
                        speech_frame_count = 0

    # ===================================================================
    # 云端识别
    # ===================================================================
    def _process_speech(self, frames):
        """将帧列表写入临时 WAV，通过 Recognition.call() 同步上传并获取结果。"""
        if not frames:
            return

        pcm_data = b"".join(frames)
        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE
        if duration_ms < 200:   # 短于 200ms 的片段忽略
            return

        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="stt_")
            os.close(fd)
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            # 先保存调试音频，再调用 API
            debug_path = "/tmp/stt_debug_last.wav"
            import shutil
            shutil.copy2(wav_path, debug_path)
            print(f"[STT] 调试音频已保存: {debug_path} ({duration_ms}ms)")

            rec = Recognition(
                model='paraformer-realtime-v2',
                format='wav',
                sample_rate=self.SAMPLE_RATE,
                callback=_DummyCallback(),
            )

            print(f"[STT] 上传语音 {duration_ms}ms 至云端...")
            result = rec.call(wav_path)

            # 调试：打印完整 result 结构
            print(f"[STT] result.output: {result.output}")
            print(f"[STT] result.get_sentence(): {result.get_sentence()}")

            sentence = result.get_sentence()
            text = sentence.get('text', '').strip() if sentence else ''

            if text and self.on_sentence_received:
                print(f"[STT] {text}")
                self.on_sentence_received(text)
            else:
                print("[STT] 云端未识别出文字")

        except Exception as e:
            print(f"[STT] 云端识别失败: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    # ===================================================================
    # VAD
    # ===================================================================
    def _vad_check(self, frame):
        try:
            return self.vad.is_speech(frame, self.SAMPLE_RATE)
        except Exception:
            return False