"""阿里云 Qwen-Omni 多模态适配器。使用 input_audio 格式拼接 base64 wav。"""
from .base import AbstractMultimodal


class AliyunMultimodal(AbstractMultimodal):
    """Qwen-Omni 音频消息格式。"""

    def build_audio_message(self, audio_b64: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "请回复这段语音。"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_b64}",
                        "format": "wav",
                    },
                },
            ],
        }
