import os
import yaml
import dashscope
from dashscope import MultiModalConversation

def test_qwen_audio():
    print("=== 测试 Qwen-Audio-Turbo 多模态大模型 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    tmp_path = "test_transcription_sync.wav"
    abs_path = os.path.abspath(tmp_path)
    file_url = "file://" + abs_path.replace("\\", "/")
    
    print(f"使用的音频: {file_url}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"audio": file_url},
                {"text": "请把这段语音转成文字，不要回答其他内容，只输出转写出来的文字。如果是一段噪音，请输出空字符串。"}
            ]
        }
    ]
    
    models = ["qwen-audio", "qwen2-audio-instruct"]
    for model_name in models:
        print(f"\n--- 测试模型: {model_name} ---")
        try:
            response = MultiModalConversation.call(
                model=model_name,
                messages=messages
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"最终文本: {response.output.choices[0].message.content[0]['text']}")
            else:
                print(f"错误: {response.code} - {response.message}")
        except Exception as e:
            print(f"测试报错: {e}")

if __name__ == "__main__":
    test_qwen_audio()
