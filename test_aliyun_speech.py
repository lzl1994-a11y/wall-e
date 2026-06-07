import os
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import asyncio
import edge_tts

async def create_test_wav():
    communicate = edge_tts.Communicate("你好，阿里云语音识别测试。", "zh-CN-XiaoxiaoNeural")
    await communicate.save("test_speech.mp3")
    # edge-tts saves as mp3, we can use ffmpeg or just pass the mp3 to DashScope!
    # DashScope supports mp3 format for paraformer-v1 if we specify format='mp3' (or we just use 'wav' if it's wav)

def test_sync():
    print("=== 测试 Recognition.call 有效语音识别 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    try:
        class DummyCb(RecognitionCallback):
            pass
            
        print("正在实例化 Recognition...")
        # Since edge-tts makes mp3, we tell Dashscope it's mp3!
        recognition = Recognition(
            model='paraformer-v1',
            format='mp3',
            sample_rate=16000,
            callback=DummyCb()
        )
        
        print("调用 recognition.call('test_speech.mp3')...")
        result = recognition.call('test_speech.mp3')
        
        print(f"原始 result 对象: {result}")
        print(f"get_sentence() 返回: {result.get_sentence()}")
        print(f"完整的 json output: {result.output}")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(create_test_wav())
    test_sync()
