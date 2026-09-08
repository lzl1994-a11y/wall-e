"""Single hardware output for foreground speech and continuous background music."""

import threading
import time

from services.audio_mixer import AudioMixer
from services.playback_service import PlaybackService


class MixingPlaybackService(PlaybackService):
    BLOCK_SEC = 0.02

    def __init__(self, *args, on_wake_complete=None, **kwargs):
        self._mix_lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._mixer = None
        self._next_device_attempt = 0.0
        self.on_wake_complete = on_wake_complete
        # The base constructor starts the worker; it waits until initialization.
        super().__init__(*args, **kwargs)
        self._mixer = AudioMixer(self.sample_rate)
        self._ready.set()

    def _submit(self, method, *args):
        with self._mix_lock:
            getattr(self._mixer, method)(*args)
        self._ready.set()

    def play(self, samples):
        if samples is not None:
            self._submit("play", samples)

    def mark_turn_end(self):
        self._submit("end_speech", "dialogue")

    def play_wake(self, samples, request_id):
        # Keep the complete wake clip and its acknowledgement marker together,
        # even when TTS publishes concurrently on another ROS topic.
        with self._mix_lock:
            self._mixer.play(samples)
            self._mixer.end_speech(("wake", request_id))
        self._ready.set()

    def play_music(self, samples):
        self._submit("play_music", samples)

    def mark_stream_end(self):
        self._submit("end_music")

    def _play_worker(self):
        while not self._stopped.is_set():
            self._ready.wait(timeout=0.05)
            if self._mixer is None:
                continue
            with self._mix_lock:
                active = self._mixer.active
                if not active:
                    self._ready.clear()
            if not active:
                self._close_stream(drain=True)
                continue
            try:
                self._play_mix_block()
            except Exception as exc:
                print(f"[Playback Service] 混音播放失败: {exc}")
                self._close_stream(drain=False)
                self._stopped.wait(self.BLOCK_SEC)
        self._close_stream(drain=True)

    def _play_mix_block(self):
        opened = self._stream is not None
        if not opened and time.monotonic() >= self._next_device_attempt:
            self._next_device_attempt = time.monotonic() + 1.0
            try:
                opened = self._ensure_stream()
            except Exception as exc:
                print(f"[Playback Service] 音频设备暂不可用: {exc}")
        latency = float(self._stream.latency) if opened else 0.0
        with self._mix_lock:
            audio, completed = self._mixer.render(
                max(1, round(self.sample_rate * self.BLOCK_SEC)), latency
            )
        try:
            if opened:
                self._stream.write(audio.reshape(-1, 1))
            else:
                self._stopped.wait(self.BLOCK_SEC)
        finally:
            # Even an unavailable speaker must release capture for this turn.
            for token in completed:
                if isinstance(token, tuple) and token[0] == "wake":
                    if self.on_wake_complete:
                        self.on_wake_complete(token[1])
                elif self.on_turn_complete:
                    self.on_turn_complete()

    def close(self):
        self._stopped.set()
        self._ready.set()
        self._worker.join(timeout=2.0)
