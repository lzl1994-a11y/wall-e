# services/stt_service.py
"""
[ZH] 阿里云 Paraformer 实时语音识别服务
     使用 dashscope.audio.asr.Recognition 流式 API (paraformer-realtime-v1)
     配合 WebRTC VAD 进行静音断句，断句后立即获取云端识别结果。
[EN] Alibaba Cloud Paraformer real-time STT service.
     Uses dashscope.audio.asr.Recognition streaming API with WebRTC VAD.
"""
import queue
import threading
import time
import yaml
import numpy as np
import sounddevice as sd
import webrtcvad

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback


# ---------------------------------------------------------------------------
# 云端识别结果回调
# ---------------------------------------------------------------------------
class _STTCallback(RecognitionCallback):
    """Recognition 流式回调：收集最终断句文本并通过 Event 通知主线程。"""

    def __init__(self):
        self.text = ""
        self.done = threading.Event()
        self._error = None

    def on_event(self, result):
        try:
            sentence = result.get_sentence()
            if sentence and 'text' in sentence:
                self.text = sentence['text']
        except Exception:
            pass

    def on_close(self):
        self.done.set()

    def on_error(self, message):
        self._error = message
        self.done.set()


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------
class STTService:
    """
    外部接口保持与旧版兼容:
        stt = STTService(config_path, on_sentence_received=callback)
        stt.start()   # 开始监听
        stt.stop()    # 停止监听
        stt.pause()   # 机器人说话时暂停（防回声）
        stt.resume()  # 恢复监听
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30                       # VAD 帧长
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2        # int16 = 2 bytes/frame
    SEND_INTERVAL_MS = 120              # 向云端发送音频的间隔
    SEND_INTERVAL_BYTES = int(SAMPLE_RATE * SEND_INTERVAL_MS / 1000) * 2
    SILENCE_SEC = 0.8                   # 静音断句阈值
    RESULT_TIMEOUT = 4.0                # 等待云端结果的超时

    def __init__(self, config_path="core/config.yaml", on_sentence_received=None):
        self.on_sentence_received = on_sentence_received
        self.is_running = False
        self.is_paused = False

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.api_key = config['ai_settings']['api_key']

        self.vad = webrtcvad.Vad(2)     # 灵敏度 0-3，2 适合桌面环境
        self.audio_queue = queue.Queue(maxsize=300)

        # 内部状态
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
        print("[STT] Paraformer 实时语音监听已启动")
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
        """
        主循环：从音频队列取帧 → VAD 检测 → 驱动 Recognition 流式会话。
        状态机：IDLE → LISTENING → WAITING_RESULT → IDLE
        """
        dashscope.api_key = self.api_key

        max_silence = int(self.SILENCE_SEC / (self.FRAME_MS / 1000.0))
        byte_buf = bytearray()
        send_buf = bytearray()

        silence_count = 0
        in_speech = False
        recognition = None
        callback = None

        while self.is_running:
            if self._paused_event.is_set():
                time.sleep(0.1)
                byte_buf.clear()
                send_buf.clear()
                in_speech = False
                if recognition:
                    self._abort_recognition(recognition)
                    recognition = None
                continue

            # 取音频数据（带超时避免空转）
            try:
                byte_buf.extend(self.audio_queue.get(timeout=0.1))
            except queue.Empty:
                pass

            # 逐帧处理
            while len(byte_buf) >= self.FRAME_BYTES:
                frame = bytes(byte_buf[:self.FRAME_BYTES])
                del byte_buf[:self.FRAME_BYTES]

                is_speech = self._vad_check(frame)

                if is_speech:
                    silence_count = 0
                    if not in_speech:
                        # 状态切换：IDLE → LISTENING
                        in_speech = True
                        send_buf.clear()
                        recognition, callback = self._start_recognition()
                        print("[STT] 检测到人声，开始流式传输")
                        # 首帧立即发送，防止 WebSocket 服务端超时断开
                        if recognition:
                            send_buf.extend(frame)
                            self._send_now(send_buf, recognition)
                        continue
                    if recognition:
                        send_buf.extend(frame)
                        self._try_flush(send_buf, recognition)
                elif in_speech:
                    silence_count += 1
                    if recognition:
                        send_buf.extend(frame)
                        self._try_flush(send_buf, recognition)
                    if silence_count > max_silence:
                        # 状态切换：LISTENING → WAITING_RESULT
                        in_speech = False
                        silence_count = 0
                        result = self._finish_recognition(
                            recognition, callback, send_buf
                        )
                        recognition = None
                        callback = None
                        if result and self.on_sentence_received:
                            print(f"[STT] {result}")
                            self.on_sentence_received(result)

    # ===================================================================
    # Recognition 会话管理
    # ===================================================================
    def _start_recognition(self):
        cb = _STTCallback()
        rec = Recognition(
            model='paraformer-realtime-v1',
            format='pcm',
            sample_rate=self.SAMPLE_RATE,
            callback=cb,
        )
        try:
            rec.start()
        except Exception as e:
            print(f"[STT] Recognition 启动失败: {e}")
            return None, None
        return rec, cb

    def _send_now(self, send_buf, recognition):
        """立即发送缓冲数据（不管是否攒够阈值），用于首帧保活。"""
        try:
            recognition.send_audio_frame(bytes(send_buf))
        except Exception as e:
            print(f"[STT] 发送音频帧失败: {e}")
        send_buf.clear()

    def _try_flush(self, send_buf, recognition):
        """攒够 SEND_INTERVAL_BYTES 后一次性发送，避免触发云端限流。"""
        if len(send_buf) >= self.SEND_INTERVAL_BYTES:
            try:
                recognition.send_audio_frame(bytes(send_buf))
            except Exception as e:
                print(f"[STT] 发送音频帧失败: {e}")
            send_buf.clear()

    def _finish_recognition(self, recognition, callback, send_buf):
        """停止 Recognition 会话并取回最终识别文本。"""
        if not recognition or not callback:
            return ""
        # 发送残留数据
        if send_buf:
            try:
                recognition.send_audio_frame(bytes(send_buf))
            except Exception:
                pass
            send_buf.clear()
        # 停止会话
        try:
            recognition.stop()
        except Exception:
            pass
        callback.done.wait(timeout=self.RESULT_TIMEOUT)
        return callback.text.strip()

    def _abort_recognition(self, recognition):
        try:
            recognition.stop()
        except Exception:
            pass

    # ===================================================================
    # VAD
    # ===================================================================
    def _vad_check(self, frame):
        try:
            return self.vad.is_speech(frame, self.SAMPLE_RATE)
        except Exception:
            return False