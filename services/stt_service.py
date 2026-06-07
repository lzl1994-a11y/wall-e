# services/stt_service.py
import queue
import threading
import time
import wave
import tempfile
import os
import yaml
import numpy as np
import sounddevice as sd
import webrtcvad
from openai import OpenAI

class STTService:
    """
    [ZH] 阿里云 SenseVoice 语音识别服务 (带 WebRTC VAD 静音检测)
         抛弃本地低效模型，利用 VAD 智能切分句子后秒级传输至云端。
    """
    def __init__(self, config_path="core/config.yaml", on_sentence_received=None):
        self.on_sentence_received = on_sentence_received
        self.is_running = False
        self.is_paused = False
        
        # 用于缓存录音的队列
        self.audio_queue = queue.Queue(maxsize=300)
        
        # 1. 读取配置文件中的 API Key 和 URL
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        api_key = config['ai_settings']['api_key']
        self.api_key = api_key
        # 注意：SenseVoice 强制要求使用兼容模式的 v1 端点
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 2. 初始化 WebRTC VAD (Voice Activity Detection)
        self.sample_rate = 16000
        # VAD 灵敏度 0-3 (3是最激进的过滤模式，只对真实人声敏感)
        self.vad = webrtcvad.Vad(3)
        
        # WebRTC 强制要求帧长为 10, 20 或 30 毫秒
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        
        self.silence_threshold = 0.8  # 当停止说话超过 0.8 秒后触发断句
        self.voice_buffer = []        # 存放当前这一句话的音频切片
        self.silence_frames = 0
        
        self._listen_thread = None
        self.audio_stream = None

    def start(self):
        self.is_running = True
        self._listen_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._listen_thread.start()

        # [内部回调] 麦克风硬件的中断回调
        def callback(indata, frames, time_info, status):
            if not self.is_running or self.is_paused:
                return
            # 将麦克风采样的 float32 数据转成 PCM 16-bit
            int16_data = (indata * 32767).astype(np.int16)
            try:
                self.audio_queue.put_nowait(int16_data.tobytes())
            except queue.Full:
                pass

        self.audio_stream = sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=self.sample_rate,
            blocksize=self.frame_size, # 每次吐出 30ms (480采样点)
            callback=callback,
        )
        self.audio_stream.start()
        print("[STT] 🎙️ 阿里云 SenseVoice 语音监听已启动...")
        return True

    def _process_loop(self):
        """核心处理线程：实时流式传输音频至云端"""
        in_speech = False
        max_silence_frames = int(self.silence_threshold / (self.frame_duration_ms / 1000.0))
        
        byte_buffer = bytearray()
        chunk_size = self.frame_size * 2 # 16-bit 占用2个字节 (30ms = 960 bytes)
        
        # 专门用于攒够 100ms (3200 bytes) 后再发送的缓冲区
        send_buffer = bytearray()
        
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback
        import threading
        dashscope.api_key = self.api_key
        
        recognition = None
        cb = None

        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                byte_buffer.clear()
                send_buffer.clear()
                continue

            try:
                # 每隔 100ms 拉取一次队列
                incoming_bytes = self.audio_queue.get(timeout=0.1)
                byte_buffer.extend(incoming_bytes)
            except queue.Empty:
                pass

            # 当缓冲里凑齐了完整的 30ms 帧，就进行一次 VAD 检测并准备流式发送
            while len(byte_buffer) >= chunk_size:
                frame_bytes = bytes(byte_buffer[:chunk_size])
                del byte_buffer[:chunk_size]

                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                except Exception:
                    continue

                if is_speech:
                    if not in_speech:
                        in_speech = True
                        self.silence_frames = 0
                        send_buffer.clear()
                        print("[STT] 🗣️ 检测到人声，开始实时流式传输...")
                        
                        class STTCallback(RecognitionCallback):
                            def __init__(self):
                                self.text = ""
                                self.done = threading.Event()
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
                                print("[STT] 阿里云流式连接结束 (已被服务器安全断开)。")
                                self.done.set()

                        cb = STTCallback()
                        recognition = Recognition(
                            model='paraformer-realtime-v1',
                            format='pcm',
                            sample_rate=16000,
                            callback=cb,
                            api_key=self.api_key # 显式传入确保子线程有权限
                        )
                        try:
                            recognition.start()
                        except Exception as e:
                            print(f"[STT] 阿里云启动失败: {e}")
                            recognition = None

                    self.silence_frames = 0
                    if recognition:
                        send_buffer.extend(frame_bytes)
                        # 必须攒够 100ms (3200 bytes) 再发送，防止 30ms (960 bytes) 发送太快触发阿里云防洪断连
                        if len(send_buffer) >= 3200:
                            try:
                                recognition.send_audio_frame(bytes(send_buffer))
                            except Exception as e:
                                print(f"[STT] ⚠️ 发送失败: {e}")
                                recognition = None
                            send_buffer.clear()

                elif in_speech:
                    self.silence_frames += 1
                    if recognition:
                        send_buffer.extend(frame_bytes)
                        if len(send_buffer) >= 3200:
                            try:
                                recognition.send_audio_frame(bytes(send_buffer))
                            except Exception as e:
                                print(f"[STT] ⚠️ 发送失败: {e}")
                                recognition = None
                            send_buffer.clear()
                    
                    # 达到了断句的阈值
                    if self.silence_frames > max_silence_frames:
                        in_speech = False
                        print("[STT] ☁️ 语音结束，正在获取最终识别结果...")
                        if recognition:
                            # 发送残留的数据
                            if len(send_buffer) > 0:
                                try:
                                    recognition.send_audio_frame(bytes(send_buffer))
                                except Exception:
                                    pass
                                send_buffer.clear()
                                
                            try:
                                recognition.stop()
                            except Exception:
                                pass
                                
                            cb.done.wait(timeout=3.0)
                            text = cb.text.strip()
                            if text and self.on_sentence_received:
                                print(f"[STT] ✅ 识别结果: {text}")
                                self.on_sentence_received(text)
                            else:
                                print("[STT] 阿里云未能识别出文字 (可能声音太小或只有底噪)。")
                            
                            recognition = None
                            cb = None

    def pause(self):
        """Pause listening while the robot speaks (自听抵消)."""
        self.is_paused = True
        self.voice_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        print("\n[STT] ⏸️ 麦克风已暂停监听 (防止回声)")

    def resume(self):
        """Resume listening with clean buffers."""
        self.voice_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.is_paused = False
        print("[STT] ▶️ 麦克风已恢复监听")

    def stop(self):
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1.0)
        print("[STT] 🛑 语音监听已完全停止.")
