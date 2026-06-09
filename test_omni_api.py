"""用已有 WAV 直接测试 Qwen-Omni（修正格式：input_audio.data + stream=True）"""
import base64
import yaml
from openai import OpenAI

with open("core/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(api_key=config["ai_settings"]["api_key"],
                base_url=config["ai_settings"]["base_url"])

with open("stt_debug_last.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

messages = [
    {"role": "system", "content": "你叫瓦力，一个毒舌的桌面机器人，回复要简短、带讽刺语气。"},
    {"role": "user", "content": [
        {"type": "text", "text": "请回复这段语音。"},
        {"type": "input_audio", "input_audio": {
            "data": f"data:;base64,{audio_b64}",
            "format": "wav",
        }},
    ]}
]

print(f"发送 WAV 到 qwen-omni-turbo ...")
try:
    r = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages=messages,
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        timeout=15,
    )
    full_text = []
    for chunk in r:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_text.append(delta.content)
    print(f"回复: {''.join(full_text)}")
except Exception as e:
    print(f"失败: {e}")