import time
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import threading

def test_sync():
    print("=== 测试 Recognition 发送满载噪音 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    pcm_data = b'\xff\x7f' * 16000 # 1 秒的满载噪音 (32000 bytes)
        
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
            print(f"[Callback Error]: {message}")
            self.done.set()

    cb = STTCallback()
    print("正在实例化 Recognition...")
    recognition = Recognition(
        model='paraformer-realtime-v1',
        format='pcm',
        sample_rate=16000,
        callback=cb
    )
    
    recognition.start()
    
    chunk_size = 3200 # 100ms
    print(f"开始发送音频，总长度 {len(pcm_data)} 字节...")
    for i in range(0, len(pcm_data), chunk_size):
        recognition.send_audio_frame(pcm_data[i:i+chunk_size])
        time.sleep(0.1)
        
    print("停止发送并等待结果...")
    recognition.stop()
    cb.done.wait(timeout=5.0)
    
    print(f"===========================")
    print(f"最终识别文字: {cb.text}")
    print(f"===========================")
        
if __name__ == "__main__":
    test_sync()
