"""Xiaomi MiMo adapter for the audio-direct pipeline."""

from .base import AbstractMultimodal


class XiaomiMiMoMultimodal(AbstractMultimodal):
    """Build MiMo-V2.5's OpenAI-compatible Base64 WAV content."""

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
                        "data": f"data:audio/wav;base64,{audio_b64}",
                    },
                },
            ],
        }
