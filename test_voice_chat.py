"""测试 voice_chat_service：直接说话 → 看 Qwen-Omni 是否正常回复"""
import sys
sys.path.insert(0, "/home/pi/wali_x3_brain")

from services.voice_chat_service import VoiceChatService

vc = VoiceChatService(config_path="core/config.yaml")
vc.on_llm_reply = lambda text: print(f"\n>>> 瓦力: {text}\n")

print("=== 直接语音对话测试 ===")
print("对着麦克风说话，说完停顿 0.8 秒自动发送")
print("按 Ctrl+C 停止\n")

vc.start()

try:
    while True:
        import time
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n停止中...")
finally:
    vc.stop()