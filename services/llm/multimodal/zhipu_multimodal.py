"""智谱多模态适配器（占位，智谱暂不支持音频输入）。"""
from .base import AbstractMultimodal


class ZhipuMultimodal(AbstractMultimodal):
    """智谱暂无音频多模态模型，占位。"""

    def build_audio_message(self, audio_b64: str) -> dict:
        raise NotImplementedError("智谱暂不支持音频多模态输入")
