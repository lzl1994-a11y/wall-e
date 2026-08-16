"""Boundary-silence normalization for clean synthesized PCM audio."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AudioTrimResult:
    samples: np.ndarray
    first_segment: bool
    leading_cut_ms: float
    trailing_cut_ms: float
    original_ms: float
    processed_ms: float


class TurnAudioTrimmer:
    """Trim TTS segment boundaries while preserving a natural turn opening."""

    def __init__(
        self,
        sample_rate=48000,
        keep_silence_ms=100.0,
        threshold_dbfs=-45.0,
        window_ms=10.0,
    ):
        self.sample_rate = int(sample_rate)
        self.keep_silence_ms = max(0.0, float(keep_silence_ms))
        self.threshold_dbfs = float(threshold_dbfs)
        self.window_ms = max(1.0, float(window_ms))
        self._first_segment = True

    def reset(self):
        self._first_segment = True

    def process(self, samples: np.ndarray) -> AudioTrimResult:
        audio = np.asarray(samples, dtype=np.int16)
        first_segment = self._first_segment
        self._first_segment = False
        original_ms = self._duration_ms(audio.size)

        bounds = self._active_bounds(audio)
        if bounds is None:
            return AudioTrimResult(
                samples=audio,
                first_segment=first_segment,
                leading_cut_ms=0.0,
                trailing_cut_ms=0.0,
                original_ms=original_ms,
                processed_ms=original_ms,
            )

        active_start, active_end = bounds
        keep_samples = int(round(self.sample_rate * self.keep_silence_ms / 1000.0))
        trim_start = 0 if first_segment else max(0, active_start - keep_samples)
        trim_end = min(audio.size, active_end + keep_samples)
        if trim_end <= trim_start:
            trim_start = 0
            trim_end = audio.size

        processed = audio[trim_start:trim_end].copy()
        return AudioTrimResult(
            samples=processed,
            first_segment=first_segment,
            leading_cut_ms=self._duration_ms(trim_start),
            trailing_cut_ms=self._duration_ms(audio.size - trim_end),
            original_ms=original_ms,
            processed_ms=self._duration_ms(processed.size),
        )

    def _active_bounds(self, samples: np.ndarray):
        if samples.size == 0:
            return None
        window_size = max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))
        window_count = int(math.ceil(samples.size / window_size))
        padded_size = window_count * window_size
        values = samples.astype(np.float32)
        if padded_size != samples.size:
            values = np.pad(values, (0, padded_size - samples.size))
        windows = values.reshape(window_count, window_size)
        rms = np.sqrt(np.mean(np.square(windows), axis=1))
        threshold = 32767.0 * (10.0 ** (self.threshold_dbfs / 20.0))
        active = np.flatnonzero(rms >= threshold)
        if active.size == 0:
            return None
        start = int(active[0]) * window_size
        end = min(samples.size, (int(active[-1]) + 1) * window_size)
        return start, end

    def _duration_ms(self, sample_count):
        return float(sample_count) * 1000.0 / self.sample_rate
