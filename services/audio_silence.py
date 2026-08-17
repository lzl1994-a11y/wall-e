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

    def mark_segment(self):
        """Advance turn state for a segment handled by a streaming trimmer."""
        first_segment = self._first_segment
        self._first_segment = False
        return first_segment

    def process(self, samples: np.ndarray) -> AudioTrimResult:
        audio = np.asarray(samples, dtype=np.int16)
        first_segment = self.mark_segment()
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
        trailing_start = max(0, active_end - trim_start)
        if trailing_start < processed.size:
            processed[trailing_start:] = 0
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


class StreamingTailSilenceTrimmer:
    """Pass speech immediately while holding only possible trailing silence."""

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
        self.window_size = max(1, int(round(self.sample_rate * window_ms / 1000.0)))
        self._threshold = 32767.0 * (10.0 ** (self.threshold_dbfs / 20.0))
        self.reset()

    def reset(self):
        self._seen_active = False
        self._window_buffer = np.array([], dtype=np.int16)
        self._pending_silence = []

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.int16)
        if audio.size == 0:
            return audio
        if self._window_buffer.size:
            audio = np.concatenate((self._window_buffer, audio))

        complete_size = audio.size - (audio.size % self.window_size)
        self._window_buffer = audio[complete_size:].copy()
        emitted = []
        for offset in range(0, complete_size, self.window_size):
            self._consume_window(audio[offset : offset + self.window_size], emitted)
        return self._join(emitted)

    def finish(self) -> np.ndarray:
        emitted = []
        if self._window_buffer.size:
            self._consume_window(self._window_buffer, emitted)
        if self._pending_silence:
            pending = np.concatenate(self._pending_silence)
            keep_samples = int(round(self.sample_rate * self.keep_silence_ms / 1000.0))
            if keep_samples:
                emitted.append(
                    np.zeros(min(pending.size, keep_samples), dtype=np.int16)
                )
        result = self._join(emitted)
        self.reset()
        return result

    def _consume_window(self, window, emitted):
        values = window.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
        if rms >= self._threshold:
            if self._pending_silence:
                emitted.extend(self._pending_silence)
                self._pending_silence = []
            emitted.append(window.copy())
            self._seen_active = True
        elif self._seen_active:
            self._pending_silence.append(window.copy())
        else:
            emitted.append(window.copy())

    @staticmethod
    def _join(parts):
        if not parts:
            return np.array([], dtype=np.int16)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)
