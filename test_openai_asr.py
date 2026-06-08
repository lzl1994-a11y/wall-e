import yaml
from openai import OpenAI

def test_openai_asr():
    print("=== 测试 OpenAI 兼容 API ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 生成一个测试文件
    import wave
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        with wave.open(tmp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00' * 32000)

    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    models_to_test = ["sensevoice-v1", "qwen3-asr-flash"]
    
    for model_name in models_to_test:
        print(f"\n尝试模型: {model_name} ...")
        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                  model=model_name,
                  file=audio_file,
                  response_format="text"
                )
            print(f"✅ {model_name} 成功! 结果: {transcription}")
        except Exception as e:
            print(f"❌ {model_name} 失败: {e}")
            
    os.remove(tmp_path)

if __name__ == "__main__":
    test_openai_asr()
