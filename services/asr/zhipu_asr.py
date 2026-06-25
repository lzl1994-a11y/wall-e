"""智谱 ASR 适配器（占位，待实现）。"""
from .base import AbstractASR


class ZhipuASR(AbstractASR):
    """智谱语音识别 —— 暂未实现，接口留空。"""

    def __init__(self, api_key: str, url: str, model: str):
        self.api_key = api_key
        self.url = url
        self.model = model

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        # TODO: 智谱 ASR API 接入
        print("[ZhipuASR] 适配器未实现")
        return ""
