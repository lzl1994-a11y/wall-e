"""多模态适配器抽象基类。将音频数据封装为各平台 LLM 能理解的消息格式。"""
from abc import ABC, abstractmethod


class AbstractMultimodal(ABC):
    """构建带音频的 messages，供 voice_chat 拼入对话上下文后直接调 LLM。"""

    @abstractmethod
    def build_audio_message(self, audio_b64: str) -> dict:
        """返回 user role 的完整消息 dict，内含 base64 音频。"""
        ...
