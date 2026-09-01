"""Tencent Hunyuan adapter for the audio-direct pipeline."""

from .base import AbstractMultimodal


class TencentHunyuanMultimodal(AbstractMultimodal):
    """Reject audio-direct requests that Hunyuan Chat cannot represent.

    Hunyuan's OpenAI-compatible Chat API supports text, images and video, but
    its documented content types do not include OpenAI ``input_audio``.  Image
    understanding is already handled by :class:`services.llm_service.LLMService`;
    voice input must therefore go through the project's ASR -> LLM pipeline.
    """

    def build_audio_message(self, audio_b64: str) -> dict:
        raise NotImplementedError(
            "腾讯混元 Chat 接口暂不支持 input_audio；"
            "请将 pipeline.mode 设置为 asr_llm，由 ASR 转写后再调用混元模型"
        )
