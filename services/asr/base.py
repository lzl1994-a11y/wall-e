"""ASR 适配器抽象基类。所有 ASR 提供商实现此接口。"""
from abc import ABC, abstractmethod


class AbstractASR(ABC):
    """wav 文件 → 识别文本"""

    @abstractmethod
    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        """上传 wav 到云端 ASR，返回识别文本。失败返回空字符串。"""
        ...
