"""诊断：指定 WAV 文件调用 Recognition.call()"""
import sys
import os
import yaml
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

with open("core/config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
dashscope.api_key = config['ai_settings']['api_key']

class CB(RecognitionCallback):
    pass

wav_path = sys.argv[1] if len(sys.argv) > 1 else "stt_debug_last.wav"
print(f"文件: {wav_path}  ({os.path.getsize(wav_path)} bytes)")

rec = Recognition(model='paraformer-realtime-v1', format='wav', sample_rate=16000, callback=CB())
result = rec.call(wav_path)
print(f"status_code  : {result.status_code}")
print(f"output       : {result.output}")
sentence = result.get_sentence()
print(f"get_sentence : {sentence}")
print(f"type         : {type(sentence)}")

if isinstance(sentence, list) and sentence:
    print(f"  sentence[0]: {sentence[0]}")
    text = sentence[0].get('text', '')
elif isinstance(sentence, dict):
    text = sentence.get('text', '')
else:
    text = ''
print(f"最终文本     : '{text}'")