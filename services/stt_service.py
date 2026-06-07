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
        """核心处理线程：智能拼装音频并调用 API"""
        in_speech = False
        max_silence_frames = int(self.silence_threshold / (self.frame_duration_ms / 1000.0))
        
        byte_buffer = bytearray()
        chunk_size = self.frame_size * 2 # 16-bit 占用2个字节

        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                byte_buffer.clear()
                continue

            try:
                # 每隔 100ms 拉取一次队列
                incoming_bytes = self.audio_queue.get(timeout=0.1)
                byte_buffer.extend(incoming_bytes)
            except queue.Empty:
                pass

            # 当缓冲里凑齐了完整的 30ms 帧，就进行一次 VAD 检测
            while len(byte_buffer) >= chunk_size:
                frame_bytes = bytes(byte_buffer[:chunk_size])
                del byte_buffer[:chunk_size]

                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if is_speech:
                    in_speech = True
                    self.silence_frames = 0
                    self.voice_buffer.append(frame_bytes)
                elif in_speech:
                    self.silence_frames += 1
                    self.voice_buffer.append(frame_bytes)
                    
                    # 达到了断句的阈值
                    if self.silence_frames > max_silence_frames:
                        in_speech = False
                        self._trigger_asr()

    def _trigger_asr(self):
        # 如果说话时间太短 (比如只有一点噪音)，直接抛弃
        if len(self.voice_buffer) < (0.4 / (self.frame_duration_ms / 1000.0)):
            self.voice_buffer = []
            return

        print("[STT] ☁️ 检测到语音结束，正在请求阿里云识别...")
        
        audio_data = b"".join(self.voice_buffer)
        self.voice_buffer = []
        self.silence_frames = 0
        
        # 封装为标准 WAV 格式临时文件发送给云端
        import tempfile
        import wave
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with wave.open(tmp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback
            
            dashscope.api_key = self.api_key
            
            # 由于 Recognition(self) 必须要求传入一个 callback，我们放一个空的
            class DummyCb(RecognitionCallback):
                pass
                
            # 实例化 Recognition，使用 paraformer-v1（非 realtime 版本，专门用来做整句短语音）
            recognition = Recognition(
                model='paraformer-v1',
                format='wav',
                sample_rate=16000,
                callback=DummyCb()
            )
            
            # 使用官方 SDK 提供的内置单文件同步上传方法（它会自动帮我们在后台分块处理，不会断开）
            result = recognition.call(tmp_path)
            
            text = ""
            sentence = result.get_sentence()
            if sentence:
                if isinstance(sentence, list):
                    text = "".join(item.get('text', '') for item in sentence)
                elif isinstance(sentence, dict):
                    text = sentence.get('text', '')
            
            text = text.strip()
            if text and self.on_sentence_received:
                print(f"[STT] ✅ 识别结果: {text}")
                self.on_sentence_received(text)
            elif not text:
                print("[STT] 阿里云未识别到有效文字 (可能是静音或噪音)。")
                
        except Exception as e:
            print(f"[STT] ❌ 阿里云调用失败: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
