"""阿里云 Paraformer ASR 适配器（dashscope SDK）。"""
import asyncio
import os

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

from .base import AbstractASR


class _DummyCallback(RecognitionCallback):
    pass


class AliyunASR(AbstractASR):
    """阿里云语音识别。配置从 config.yaml 的 asr 节读取。"""

    def __init__(self, api_key: str, model: str, url: str = ""):
        self.api_key = api_key
        self.model = model
        # url 由 DashScope SDK 内部管理，此处仅兼容工厂传参，不实际使用
        self._init_done = False

    def _ensure_loop(self):
        """Linux 后台线程无 asyncio 事件循环，dashscope SDK 依赖它。"""
        if not self._init_done:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            dashscope.api_key = self.api_key
            self._init_done = True

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        self._ensure_loop()
        try:
            rec = Recognition(
                model=self.model,
                format="wav",
                sample_rate=sample_rate,
                callback=_DummyCallback(),
            )
            result = rec.call(wav_path)
            sentence = result.get_sentence()
            if isinstance(sentence, list) and sentence:
                sentence = sentence[0]
            text = sentence.get("text", "").strip() if isinstance(sentence, dict) else ""
            return text
        except Exception as e:
            print(f"[AliyunASR] 识别失败: {e}")
            return ""
