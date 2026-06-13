# services/tts_service.py
"""TTS 合成服务：文本 → edge-tts → PCM int16 numpy array。

纯合成，不负责播放。播放由 playback_service 统一管理。
"""

import asyncio
import io
import threading

import edge_tts
import numpy as np


class TTSService:
    """Edge-TTS 合成器：文本 → PCM int16 16kHz mono。"""

    def __init__(self, voice="zh-CN-YunxiaNeural", rate="+20%", pitch="+5Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

        # 后台 event loop（edge-tts 需要异步调用）
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        print(f"[TTS Service] 合成器就绪 (voice={voice}, rate={rate}, pitch={pitch})")

    def synthesize(self, text: str) -> np.ndarray:
        """同步接口：文本 → PCM int16 数组（16kHz mono）。"""
        if not text or not text.strip():
            raise ValueError("text is empty")

        future = asyncio.run_coroutine_threadsafe(
            self._download(text), self._loop
        )
        return future.result()

    async def _download(self, text: str) -> np.ndarray:
        """异步下载 MP3 → pydub 解码 → PCM int16 16kHz mono。"""
        communicate = edge_tts.Communicate(
            text, self.voice, rate=self.rate, pitch=self.pitch
        )
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]

        if not mp3_data:
            raise RuntimeError("edge-tts 返回空音频")

        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        pcm = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return np.array(pcm.get_array_of_samples(), dtype=np.int16)

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
