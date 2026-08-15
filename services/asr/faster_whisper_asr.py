"""faster-whisper 本地离线 ASR 适配器。"""

from __future__ import annotations

import wave
from pathlib import Path

from .base import AbstractASR


class FasterWhisperASR(AbstractASR):
    def __init__(
        self,
        *,
        model_path: str,
        language: str = "zh",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        resolved_model_path = Path(model_path).expanduser().resolve()
        if not resolved_model_path.is_dir():
            raise FileNotFoundError(f"model_path 模型目录不存在: {resolved_model_path}")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Faster-Whisper ASR 需要安装 faster-whisper，请先执行: pip install faster-whisper"
            ) from exc

        self.model_path = str(resolved_model_path)
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model = WhisperModel(
            self.model_path,
            device=device,
            compute_type=compute_type,
        )

    @staticmethod
    def _validate_wav(wav_path: str, sample_rate: int) -> None:
        with wave.open(str(Path(wav_path)), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            wav_rate = wav_file.getframerate()
        if channels != 1 or sample_width != 2 or wav_rate != 16000 or sample_rate != 16000:
            raise ValueError(
                "Local ASR requires mono, 16-bit, 16 kHz PCM WAV audio "
                f"(got channels={channels}, width={sample_width}, rate={wav_rate})"
            )

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        try:
            self._validate_wav(wav_path, sample_rate)
            language = None if self.language == "auto" else self.language
            segments, _ = self._model.transcribe(
                wav_path,
                language=language,
                task="transcribe",
            )
            return "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            print(f"[FasterWhisperASR] 识别失败: {exc}")
            return ""
