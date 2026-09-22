"""Volcano Engine Ark / Doubao multimodal audio message adapter."""

from .base import AbstractMultimodal


class DoubaoMultimodal(AbstractMultimodal):
    """Build the Ark Chat Completions ``input_audio`` content shape.

    Ark expects raw Base64 data in ``input_audio.data`` and the encoding name
    separately in ``input_audio.format``.  This has been verified with
    ``doubao-seed-2-0-lite-260428`` and a 16 kHz WAV input.
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
