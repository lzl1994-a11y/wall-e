"""Two PCM lanes on one clock, with speech-driven music ducking.

The caller owns synchronization and the output device. End notifications are
delayed by output latency so a continuously playing music bed does not prevent
the dialogue from completing after its last audible sample.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _End:
    token: object = None


class _Lane:
    def __init__(self):
        self.parts = deque()
        self.active = False

    def push(self, samples):
        audio = np.asarray(samples, dtype=np.int16).reshape(-1)
        if audio.size:
            self.parts.append(audio.astype(np.float32) / 32768.0)

    def read(self, count):
        output = np.zeros(count, dtype=np.float32)
        offset = 0
        ended = None
        while self.parts and offset < count:
            part = self.parts[0]
            if isinstance(part, _End):
                ended = self.parts.popleft()
                self.active = False
                break
            self.active = True
            take = min(count - offset, part.size)
            output[offset:offset + take] = part[:take]
            offset += take
            if take == part.size:
                self.parts.popleft()
            else:
                self.parts[0] = part[take:]
        return output, offset, ended


class AudioMixer:
    def __init__(self, sample_rate=48000, duck_gain=0.2):
        self.sample_rate = sample_rate
        self.speech = _Lane()
        self.music = _Lane()
        self.duck_gain = float(duck_gain)
        self.music_gain = 1.0
        self._tail_frames = 0
        self._pending_end = None

    @property
    def active(self):
        return bool(self.speech.parts or self.music.parts or self.speech.active
                    or self.music.active or self._pending_end is not None)

    def play(self, samples):
        self.speech.push(samples)

    def end_speech(self, token="dialogue"):
        self.speech.parts.append(_End(token))

    def play_music(self, samples):
        self.music.push(samples)

    def end_music(self):
        self.music.parts.append(_End())

    def render(self, frames, output_latency=0.0):
        """Return a float32 mono block and end tokens to notify AFTER writing it."""
        completed = []
        speech_was_active = self.speech.active
        if self._pending_end is not None:
            voice = np.zeros(frames, dtype=np.float32)
            voice_count = 0
            self._tail_frames -= frames
            if self._tail_frames <= 0:
                completed.append(self._pending_end.token)
                self._pending_end = None
        else:
            voice, voice_count, end = self.speech.read(frames)
            if end is not None:
                self._pending_end = end
                self._tail_frames = max(1, int(self.sample_rate * (max(0.0, output_latency) + 0.1)))

        music, _, _ = self.music.read(frames)
        duck = (voice_count > 0 or speech_was_active or self.speech.active
                or self._pending_end is not None)
        target = self.duck_gain if duck else 1.0
        # Smooth attack (30 ms) and release (300 ms), independent of chunk sizes.
        duration = 0.03 if target < self.music_gain else 0.3
        step = (1.0 - self.duck_gain) / (duration * self.sample_rate)
        direction = 1.0 if target >= self.music_gain else -1.0
        gains = self.music_gain + direction * step * np.arange(1, frames + 1)
        gains = np.clip(gains, min(self.music_gain, target), max(self.music_gain, target))
        self.music_gain = float(gains[-1])
        return np.clip(voice + music * gains, -1.0, 1.0).astype(np.float32), completed
