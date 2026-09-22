"""Small PCM buffers used to absorb cloud TTS delivery jitter."""

import numpy as np


class StreamingPCMPrebuffer:
    """Hold the beginning of a PCM stream until a playback-safe lead exists."""

    def __init__(self, sample_rate=48000, prebuffer_ms=400.0):
        self.sample_rate = int(sample_rate)
        self.prebuffer_ms = max(0.0, float(prebuffer_ms))
        self.target_samples = int(
            round(self.sample_rate * self.prebuffer_ms / 1000.0)
        )
        self.reset()

    def reset(self):
        self._started = False
        self._parts = []
        self._sample_count = 0

    def push(self, samples):
        audio = np.asarray(samples, dtype=np.int16)
        if audio.size == 0:
            return audio, False
        if self._started:
            return audio, False

        self._parts.append(audio.copy())
        self._sample_count += audio.size
        if self._sample_count < self.target_samples:
            return np.array([], dtype=np.int16), False

        self._started = True
        return self._take_buffered(), True

    def finish(self):
        if self._started or not self._parts:
            return np.array([], dtype=np.int16), False
        self._started = True
        return self._take_buffered(), True

    def _take_buffered(self):
        if len(self._parts) == 1:
            buffered = self._parts[0]
        else:
            buffered = np.concatenate(self._parts)
        self._parts = []
        self._sample_count = 0
        return buffered
