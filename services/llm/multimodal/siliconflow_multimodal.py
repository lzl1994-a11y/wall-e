"""SiliconFlow adapter for the audio-direct pipeline."""

from .base import AbstractMultimodal


class SiliconFlowMultimodal(AbstractMultimodal):
    """Build SiliconFlow's ``audio_url`` content for audio-capable models.

    SiliconFlow exposes several model families behind one OpenAI-compatible
    endpoint.  Audio capability therefore belongs to the selected model, not
    to the provider as a whole.  Known text/vision-only Kimi models are rejected
    locally with an actionable message; other models receive the documented
    ``audio_url`` data-URL shape.
    """

    _TEXT_VISION_ONLY_MODELS = frozenset({"kimi-k2.6"})

    def __init__(self, model: str = "") -> None:
        self.model = str(model or "").strip()

    def build_audio_message(self, audio_b64: str) -> dict:
        model_name = self.model.rsplit("/", 1)[-1].lower()
        if model_name in self._TEXT_VISION_ONLY_MODELS:
            raise NotImplementedError(
                f"SiliconFlow 模型 {self.model!r} 只支持文本和图片输入，不支持音频直连；"
                "请将 pipeline.mode 设置为 asr_llm，由 ASR 转写后再调用该模型"
            )

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
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_b64}",
                    },
                },
            ],
        }
