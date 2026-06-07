import os
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import urllib.request

def test_sync():
    print("=== 测试 Recognition.call 真实语音 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 下载一个公开的中文测试音频
    url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    urllib.request.urlretrieve(url, "test.wav")

    try:
        class DummyCb(RecognitionCallback):
            pass
            
        print("正在实例化 Recognition...")
        recognition = Recognition(
            model='paraformer-realtime-v1',
            format='wav',
            sample_rate=16000,
            callback=DummyCb()
        )
        
        print("调用 recognition.call('test.wav')...")
        result = recognition.call("test.wav")
        
        print(f"===========================")
        print(f"完整 result: {result}")
        print(f"get_sentence: {result.get_sentence()}")
        print(f"===========================")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_sync()
