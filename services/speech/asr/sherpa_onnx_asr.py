"""sherpa-onnx 本地离线 ASR 适配器。"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from .base import AbstractASR


def _load_sherpa_onnx():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "本地 Sherpa ASR 需要安装 sherpa-onnx，请先执行: pip install sherpa-onnx"
        ) from exc
    return sherpa_onnx


def _require_file(path: str, field: str) -> str:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"{field} 文件不存在: {model_path}")
    return str(model_path)


def _read_wav_samples(wav_path: str, sample_rate: int) -> np.ndarray:
    with wave.open(str(Path(wav_path)), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        wav_rate = wav_file.getframerate()
        if channels != 1 or sample_width != 2 or wav_rate != 16000 or sample_rate != 16000:
            raise ValueError(
                "Local ASR requires mono, 16-bit, 16 kHz PCM WAV audio "
                f"(got channels={channels}, width={sample_width}, rate={wav_rate})"
            )
        pcm = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        text = result.get("text", "")
    else:
        text = getattr(result, "text", "")
    return text.strip() if isinstance(text, str) else ""


class _SherpaOfflineASR(AbstractASR):
    def __init__(self, recognizer: Any):
        self._recognizer = recognizer

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        try:
            samples = _read_wav_samples(wav_path, sample_rate)
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self._recognizer.decode_stream(stream)
            return _result_text(self._recognizer.get_result(stream))
        except Exception as exc:
            print(f"[{type(self).__name__}] 识别失败: {exc}")
            return ""


class SherpaZipformerASR(_SherpaOfflineASR):
    def __init__(
        self,
        *,
        encoder: str,
        decoder: str,
        joiner: str,
        tokens: str,
        num_threads: int = 2,
    ):
        sherpa_onnx = _load_sherpa_onnx()
        self.encoder = _require_file(encoder, "encoder")
        self.decoder = _require_file(decoder, "decoder")
        self.joiner = _require_file(joiner, "joiner")
        self.tokens = _require_file(tokens, "tokens")
        self.num_threads = num_threads
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self.encoder,
            decoder=self.decoder,
            joiner=self.joiner,
            tokens=self.tokens,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
        )
        super().__init__(recognizer)


class SherpaParaformerASR(_SherpaOfflineASR):
    def __init__(self, *, model: str, tokens: str, num_threads: int = 2):
        sherpa_onnx = _load_sherpa_onnx()
        self.model = _require_file(model, "model")
        self.tokens = _require_file(tokens, "tokens")
        self.num_threads = num_threads
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=self.model,
            tokens=self.tokens,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
        )
        super().__init__(recognizer)


class SherpaSenseVoiceASR(_SherpaOfflineASR):
    def __init__(
        self,
        *,
        model: str,
        tokens: str,
        language: str = "auto",
        use_itn: bool = True,
        num_threads: int = 2,
    ):
        sherpa_onnx = _load_sherpa_onnx()
        self.model = _require_file(model, "model")
        self.tokens = _require_file(tokens, "tokens")
        self.language = language
        self.use_itn = use_itn
        self.num_threads = num_threads
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self.model,
            tokens=self.tokens,
            language=language,
            use_itn=use_itn,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
        )
        super().__init__(recognizer)


class SherpaWhisperASR(_SherpaOfflineASR):
    def __init__(
        self,
        *,
        encoder: str,
        decoder: str,
        tokens: str,
        language: str = "zh",
        num_threads: int = 2,
    ):
        sherpa_onnx = _load_sherpa_onnx()
        self.encoder = _require_file(encoder, "encoder")
        self.decoder = _require_file(decoder, "decoder")
        self.tokens = _require_file(tokens, "tokens")
        self.language = language
        self.num_threads = num_threads
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=self.encoder,
            decoder=self.decoder,
            tokens=self.tokens,
            language=language,
            task="transcribe",
            num_threads=num_threads,
        )
        super().__init__(recognizer)
