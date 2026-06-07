import os
import yaml
import time
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import threading
import edge_tts
import asyncio

async def create_speech(text, filename):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(filename)

def test_streaming():
    print("=== 测试 真实流式上传 (edge-tts) ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    filename = "test_edge.mp3"
    print("正在使用 edge-tts 生成语音...")
    asyncio.run(create_speech("测试一下阿里云的流式语音识别，看看能不能成功识别这句长句子。", filename))
    print(f"语音生成完毕，大小: {os.path.getsize(filename)} bytes")

    try:
        with open(filename, "rb") as f:
            mp3_data = f.read()

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
        # 传入 mp3 格式！
        recognition = Recognition(
            model='paraformer-realtime-v1',
            format='mp3',
            sample_rate=16000,
            callback=cb
        )
        
        recognition.start()
        
        # 对于 MP3，依然按字节分块发送，模拟流式。
        # MP3 数据较小，我们每次发 1024 字节，休眠 0.1 秒
        chunk_size = 1024
        print(f"开始流式发送 {len(mp3_data)} 字节...")
        for i in range(0, len(mp3_data), chunk_size):
            recognition.send_audio_frame(mp3_data[i:i+chunk_size])
            time.sleep(0.1)
            
        print("发送完毕，等待结果...")
        recognition.stop()
        cb.done.wait(timeout=5.0)
        
        print(f"===========================")
        print(f"最终识别文字: {cb.text}")
        print(f"===========================")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_streaming()
