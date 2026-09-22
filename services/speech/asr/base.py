"""ASR 适配器抽象基类。所有 ASR 提供商实现此接口。"""
from abc import ABC, abstractmethod


class AbstractASR(ABC):
    """统一的 WAV 文件到识别文本接口。"""

    supports_streaming = False

    @abstractmethod
    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        """识别单声道 PCM WAV，返回文本；运行时失败返回空字符串。"""
        ...

    def start_stream(self, sample_rate: int = 16000) -> None:
        """开始可选的实时音频会话。批处理适配器不实现此接口。"""
        raise NotImplementedError

    def accept_audio(self, pcm_data: bytes) -> None:
        """向实时会话追加单声道 PCM16 音频。"""
        raise NotImplementedError

    def finish_stream(self) -> str:
        """结束实时会话并返回一次最终文本。"""
        raise NotImplementedError

    def cancel_stream(self) -> None:
        """取消实时会话；批处理适配器无需处理。"""

    def warmup(self) -> None:
        """提前准备模型或网络连接；默认适配器无需额外处理。"""

    def close(self) -> None:
        """释放适配器持有的连接或运行时资源。"""
        self.cancel_stream()
