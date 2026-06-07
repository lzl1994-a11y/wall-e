import os
import yaml
import tempfile
import wave
from openai import OpenAI

def test_openai_audio():
    print("=== 测试 OpenAI 兼容模式语音识别 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        api_key = config['ai_settings']['api_key']
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
        import requests
        print("正在发送伪造音频给阿里云 OpenAI 兼容接口...")
        with open(tmp_path, "rb") as f:
            files = {
                'file': ('test.wav', f, 'audio/wav')
            }
            data = {
                'model': 'sensevoice-v1'
            }
            headers = {
                'Authorization': f'Bearer {api_key}'
            }
            
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data
            )
            
        print(f"服务器状态码: {response.status_code}")
        print(f"服务器返回: {response.text}")
        if response.status_code == 200:
            print("✅ OpenAI 兼容接口测试成功！")
        else:
            print("❌ OpenAI 兼容接口测试失败。")
            
    except Exception as e:
        print(f"❌ 测试报错: {e}")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_openai_audio()
