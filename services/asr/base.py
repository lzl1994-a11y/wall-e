"""ASR 适配器抽象基类。所有 ASR 提供商实现此接口。"""
from abc import ABC, abstractmethod


class AbstractASR(ABC):
    """统一的 WAV 文件到识别文本接口。"""

    @abstractmethod
    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        """识别单声道 PCM WAV，返回文本；运行时失败返回空字符串。"""
        ...
