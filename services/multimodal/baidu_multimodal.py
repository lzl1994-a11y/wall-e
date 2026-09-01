"""Baidu Qianfan multimodal audio message adapter."""

from .base import AbstractMultimodal


class BaiduMultimodal(AbstractMultimodal):
    """Build the Qianfan OpenAI-compatible ``input_audio`` message.

    Qianfan's audio-capable models accept Base64 audio as an input content
    item.  The project records 16 kHz mono WAV, so no provider-side format
    conversion is required here.
    """

    def build_audio_message(self, audio_b64: str) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "请仅理解随附音频中的用户语音。direct_answer.heard_text "
                        "必须转写音频内容，不得抄写本条提示文字。"
                    ),
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                },
            ],
        }
