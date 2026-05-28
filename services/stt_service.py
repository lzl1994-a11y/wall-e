# services/stt_service.py
import sherpa_onnx
import sounddevice as sd
import threading
import queue
import time
import sys

class STTService:
    """
    瓦力语言听觉神经 (语音转文字服务)
    特性：带 VAD 断句检测，后台非阻塞运行，支持随时暂停/恢复监听 (防止自己听到自己说话)
    """
    def __init__(self, model_dir="F:\well-e-bot\sherpa-onnx", on_sentence_received=None):
        self.model_dir = model_dir
        self.on_sentence_received = on_sentence_received
        self.is_running = False
        self.is_paused = False # 用于在瓦力自己说话时暂停拾音
        
        print("⏳ [STT] 正在加载本地语音识别引擎...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{self.model_dir}/tokens.txt",
            encoder=f"{self.model_dir}/encoder-epoch-99-avg-1.int8.onnx",
            decoder=f"{self.model_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{self.model_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            # 👇 核心修改：在这里挂载你的热词本和激励分数！
            hotwords_file=f"{self.model_dir}/hotwords.txt",
            hotwords_score=2.5
        )
        self.stream = self.recognizer.create_stream()
        self._listen_thread = None

    def start(self):
        self.is_running = True
        
        # 1. 开启后台解析线程
        self._listen_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._listen_thread.start()
        
        # 2. 开启麦克风硬件音频流
        self.audio_stream = sd.InputStream(
            channels=1, 
            dtype="float32", 
            samplerate=16000, 
            callback=self._audio_callback
        )
        self.audio_stream.start()
        print("🟢 [STT] 语音监听服务已启动！")
        return True

    def _audio_callback(self, indata, frames, time_info, status):
        """底层麦克风中断回调，极其迅速地把声音塞进缓冲区"""
        if self.is_running and not self.is_paused:
            samples = indata.reshape(-1)
            self.stream.accept_waveform(16000, samples)

    def _process_loop(self):
        """后台死循环：疯狂解码，并判断是否说完了"""
        last_text = ""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            while self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
                
            # 实时获取当前识别的内容
            current_text = self.recognizer.get_result(self.stream)
            
            # 终端流式打印特效 (可选，让你看到他在听)
            if current_text != last_text and current_text:
                sys.stdout.write(f"\r👂 [瓦力倾听中]: {current_text}")
                sys.stdout.flush()
                last_text = current_text

            # 🚀 核心逻辑：判断用户是否停顿了 (默认 1.2 秒不说话触发)
            if self.recognizer.is_endpoint(self.stream):
                if current_text.strip():
                    print("") # 换行
                    # 触发回调函数，把一整句话扔给大脑！
                    if self.on_sentence_received:
                        self.on_sentence_received(current_text.strip())
                
                # 重置底层流，准备听下一句话
                self.recognizer.reset(self.stream)
                last_text = ""
                
            time.sleep(0.01)

    def pause(self):
        """暂停监听 (瓦力说话时调用)"""
        self.is_paused = True
        
    def resume(self):
        """恢复监听"""
        # 恢复前清空一下流，防止把之前的杂音带进来
        self.recognizer.reset(self.stream) 
        self.is_paused = False

    def stop(self):
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        print("🛑 [STT] 语音监听服务已关闭。")