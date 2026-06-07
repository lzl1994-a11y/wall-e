import os
import yaml
import tempfile
import wave
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

def test_sync():
    print("=== 测试 Recognition.call 同步本地文件识别 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 生成一个假的 1 秒静音 wav 文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        with wave.open(tmp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00' * 32000)

    try:
        class DummyCb(RecognitionCallback):
            pass
            
        print("正在实例化 Recognition...")
        recognition = Recognition(
            model='paraformer-v1',
            format='wav',
            sample_rate=16000,
            callback=DummyCb()
        )
        
        print("调用 recognition.call(tmp_path)...")
        result = recognition.call(tmp_path)
        print(f"调用成功！返回文本: {result.get_sentence()}")
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_sync()
