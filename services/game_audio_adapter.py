"""Adapter from libretro stereo PCM to the existing playback service."""

from __future__ import annotations

import ctypes

import numpy as np


class GamePlaybackAdapter:
    """Feed game PCM to a PlaybackService without owning any audio device."""

    def __init__(self, playback, *, gain: float = 0.4) -> None:
        self._playback = playback
        self._gain = min(1.0, max(0.0, float(gain)))

    def push_batch(self, samples: ctypes.POINTER(ctypes.c_short), frames: int) -> None:
        if not samples or frames <= 0:
            return
        stereo = np.ctypeslib.as_array(samples, shape=(frames * 2,)).reshape(-1, 2)
        mixed = stereo.astype(np.int32).sum(axis=1) // 2
        if self._gain != 1.0:
            mixed = np.rint(mixed * self._gain)
        self._playback.play(np.clip(mixed, -32768, 32767).astype(np.int16))

    def push_sample(self, left: int, right: int) -> None:
        frame = (ctypes.c_short * 2)(left, right)
        self.push_batch(frame, 1)

    def close(self) -> None:
        self._playback.mark_turn_end()


__all__ = ["GamePlaybackAdapter"]
