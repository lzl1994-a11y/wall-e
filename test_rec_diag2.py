"""诊断：WAV 文件调用 ASR — 双路径测试（dashscope SDK + requests 直调）"""
import sys
import os
import yaml

with open("core/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
api_key = config['ai_settings']['api_key']

wav_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/stt_debug_last.wav"
print(f"=== 文件: {wav_path}  ({os.path.getsize(wav_path)} bytes) ===")

# ---------- 方式 A: dashscope SDK (Recognition.call) ----------
print("\n--- A: dashscope SDK Recognition.call() ---")
try:
    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback

    class CB(RecognitionCallback):
        pass

    dashscope.api_key = api_key
    rec = Recognition(model='paraformer-realtime-v1', format='wav', sample_rate=16000, callback=CB())
    result = rec.call(wav_path)
    print(f"  status_code : {result.status_code}")
    print(f"  output      : {result.output}")
    sentence = result.get_sentence()
    print(f"  get_sentence: {sentence}")
    if isinstance(sentence, list) and sentence:
        text = sentence[0].get('text', '')
    elif isinstance(sentence, dict):
        text = sentence.get('text', '')
    else:
        text = ''
    print(f"  最终文本    : '{text}'")
except Exception as e:
    print(f"  失败: {e}")

# ---------- 方式 B: requests 直调 OpenAI 兼容端点 ----------
print("\n--- B: requests OpenAI 兼容端点 ---")
try:
    import requests
    from urllib3.filepost import choose_boundary

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions"

    # 尝试多个模型
    for model in ["paraformer-v1", "qwen3-asr-flash", "sensevoice-v1"]:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model": model},
                timeout=15,
            )
        print(f"  model={model:20s}  status={resp.status_code}  body={resp.text[:120]}")
        if resp.status_code == 200:
            break
except Exception as e:
    print(f"  失败: {e}")

# ---------- 方式 C: requests 直调 DashScope 原生 Recognition REST API ----------
print("\n--- C: requests DashScope 原生 API ---")
try:
    import requests
    import json

    # DashScope Recognition REST API (非流式文件上传)
    # https://help.aliyun.com/zh/model-studio/getting-started/models
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/recognition"

    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/octet-stream",
                "X-DashScope-DataInspection": "enable",
            },
            params={
                "model": "paraformer-realtime-v1",
                "format": "wav",
                "sample_rate": "16000",
            },
            data=f.read(),
            timeout=15,
        )
    print(f"  status={resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"  response: {json.dumps(data, ensure_ascii=False)[:300]}")
        # 从 output 中提取文本
        output = data.get("output", {})
        sentence = output.get("sentence", {})
        text = sentence.get("text", "") if sentence else ""
        print(f"  最终文本: '{text}'")
    else:
        print(f"  body: {resp.text[:200]}")
except Exception as e:
    print(f"  失败: {e}")