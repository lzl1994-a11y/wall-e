"""诊断：生成标准 WAV 测试各 API 路径"""
import os
import struct
import wave
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, Transcription
import threading

with open("core/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
dashscope.api_key = config['ai_settings']['api_key']

# 生成标准 16kHz mono 16-bit WAV（1秒 440Hz 正弦波）
SAMPLE_RATE = 16000
DURATION = 1.0
FREQ = 440
samples = int(SAMPLE_RATE * DURATION)
pcm = b""
for i in range(samples):
    val = int(16000 * __import__('math').sin(2 * 3.14159 * FREQ * i / SAMPLE_RATE))
    pcm += struct.pack('<h', max(-32768, min(32767, val)))

wav_path = "_test_sine.wav"
with wave.open(wav_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(pcm)

print(f"生成测试 WAV: {wav_path} ({os.path.getsize(wav_path)} bytes)")

# --- 用不同 API 路径测试 ---
class CB(RecognitionCallback):
    pass

# 路径 1: Recognition.call() 直接传文件路径
print("\n=== Recognition.call(path) ===")
for m in ["paraformer-realtime-v2", "paraformer-realtime-v1"]:
    print(f"\n--- {m} ---")
    try:
        rec = Recognition(model=m, format='wav', sample_rate=SAMPLE_RATE, callback=CB())
        result = rec.call(wav_path)
        print(f"status     : {result.status_code}")
        print(f"get_sentence: {result.get_sentence()}")
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")

# 路径 2: Recognition 流式 (start/send/stop) 发 PCM 数据
print("\n=== Recognition 流式 (start/send/stop) ===")
for m in ["paraformer-realtime-v2", "paraformer-realtime-v1"]:
    print(f"\n--- {m} ---")
    cb = CB()
    cb.done = threading.Event()
    cb.text = ""
    def on_event(result):
        try:
            s = result.get_sentence()
            if s and 'text' in s:
                cb.text = s['text']
        except: pass
    def on_close():
        cb.done.set()
    def on_error(msg):
        print(f"  on_error: {msg}")
        cb.done.set()
    cb.on_event = on_event
    cb.on_close = on_close
    cb.on_error = on_error
    
    try:
        rec = Recognition(model=m, format='pcm', sample_rate=SAMPLE_RATE, callback=cb)
        rec.start()
        # 分块发送 PCM 数据
        chunk = 3200
        for i in range(0, len(pcm), chunk):
            rec.send_audio_frame(pcm[i:i+chunk])
        rec.stop()
        cb.done.wait(timeout=5.0)
        print(f"text       : '{cb.text}'")
    except Exception as e:
        print(f"异常: {type(e).__name__}: {e}")

os.remove(wav_path)