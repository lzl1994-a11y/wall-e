"""诊断脚本：测试阿里云 Recognition 流式 API 是否可用"""
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
import threading

with open("core/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
dashscope.api_key = config['ai_settings']['api_key']

class CB(RecognitionCallback):
    def __init__(self):
        self.done = threading.Event()
        self.text = ""
        self.err = None
    def on_event(self, result):
        try:
            s = result.get_sentence()
            if s and 'text' in s:
                self.text = s['text']
        except:
            pass
    def on_close(self):
        print("[CB] on_close")
        self.done.set()
    def on_error(self, msg):
        print(f"[CB] on_error: {msg}")
        self.err = msg
        self.done.set()

models = ["paraformer-realtime-v2", "paraformer-realtime-v1"]
for m in models:
    print(f"\n=== 测试模型: {m} ===")
    cb = CB()
    rec = Recognition(model=m, format='pcm', sample_rate=16000, callback=cb)
    try:
        rec.start()
        print(f"[OK] start() 返回")
    except Exception as e:
        print(f"[FAIL] start() 异常: {e}")
        continue

    # 发送 500ms 静音 PCM 数据
    silence = b'\x00\x00' * 8000
    try:
        rec.send_audio_frame(silence)
        print(f"[OK] send_audio_frame() 返回")
    except Exception as e:
        print(f"[FAIL] send_audio_frame() 异常: {e}")

    try:
        rec.stop()
    except:
        pass

    cb.done.wait(timeout=5.0)
    if cb.err:
        print(f"[FAIL] 服务端错误: {cb.err}")
    else:
        print(f"[RESULT] 识别文本: '{cb.text}'")