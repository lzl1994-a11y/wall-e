"""Single hardware output for foreground speech and continuous background music."""

import threading
import time
import sys

import numpy as np
import sounddevice as sd

from services.audio.audio_mixer import AudioMixer
from services.audio.playback_service import PlaybackService


class MixingPlaybackService(PlaybackService):
    BLOCK_SEC = 0.02
    OUTPUT_LATENCY_SEC = 0.08

    def __init__(self, *args, on_wake_complete=None, on_system_complete=None, **kwargs):
        # Device "low" latency can be shorter than a 20 ms write block. On
        # the X3, scheduler pauses repeatedly exhausted that hardware buffer.
        kwargs.setdefault("latency", self.OUTPUT_LATENCY_SEC)
        self._mix_lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._mixer = None
        self._next_device_attempt = 0.0
        self._interrupt_requested = threading.Event()
        self._output_underflows = 0
        self._realtime_started_at = None
        self._realtime_underflow_base = 0
        self._echo_reference = None
        self._speaker_gain = 1.0
        self.on_wake_complete = on_wake_complete
        self.on_system_complete = on_system_complete
        # The base constructor starts the worker; it waits until initialization.
        super().__init__(*args, **kwargs)
        self._mixer = AudioMixer(self.sample_rate)
        if sys.platform == "linux" and self.sample_rate == 48000:
            from services.audio.echo_reference import EchoReferenceSender
            self._echo_reference = EchoReferenceSender()
        self._ready.set()

    def _submit(self, method, *args):
        with self._mix_lock:
            getattr(self._mixer, method)(*args)
        self._ready.set()

    def play(self, samples):
        if samples is not None:
            self._submit("play", samples)

    def play_realtime(self, samples, max_buffer_sec=0.12):
        if samples is not None:
            with self._mix_lock:
                if self._mixer.realtime_stats is None:
                    self._realtime_started_at = time.monotonic()
                    self._realtime_underflow_base = self._output_underflows
                self._mixer.play_realtime(
                    samples, max(1, round(self.sample_rate * float(max_buffer_sec)))
                )
            self._ready.set()

    def mark_turn_end(self):
        self._submit("end_speech", "dialogue")

    def stop_speech(self):
        """Abort foreground speech on a barge-in while keeping music alive."""
        with self._mix_lock:
            if self._mixer is not None:
                stats = self._mixer.realtime_stats
                if stats is not None:
                    elapsed = time.monotonic() - self._realtime_started_at
                    print("[Playback Service] 实时对讲播放统计: "
                          f"seconds={elapsed:.3f} "
                          + " ".join(f"{key}={value}" for key, value in stats.items())
                          + f" output_underflows={self._output_underflows - self._realtime_underflow_base}",
                          flush=True)
                self._mixer.stop_speech()
        self._interrupt_requested.set()
        self._ready.set()

    def play_wake(self, samples, request_id):
        self.play_prompt(samples, "wake", request_id)

    def play_prompt(self, samples, category, request_id):
        # Keep each complete prompt and its acknowledgement marker together,
        # even when TTS publishes concurrently on another ROS topic.
        with self._mix_lock:
            self._mixer.play(samples)
            self._mixer.end_speech(("prompt", category, request_id))
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
        if self._interrupt_requested.is_set():
            self._interrupt_requested.clear()
            self._close_stream(drain=False)
            return
        opened = self._stream is not None
        if not opened and time.monotonic() >= self._next_device_attempt:
            try:
                opened = self._ensure_stream()
            except Exception as exc:
                print(f"[Playback Service] 音频设备暂不可用: {exc}")
            # Back off only an unavailable device, never a successful open.
            # A new PTT turn must be playable immediately after an abort.
            self._next_device_attempt = 0.0 if opened else time.monotonic() + 1.0
            if opened:
                print(f"[Playback Service] 混音输出实际延迟: {self._stream.latency:.3f}s", flush=True)
                if self._echo_reference:
                    self._echo_reference.configure_device(sd.query_devices(self._device, "output")["name"])
        latency = float(self._stream.latency) if opened else 0.0
        with self._mix_lock:
            audio, completed = self._mixer.render(
                max(1, round(self.sample_rate * self.BLOCK_SEC)), latency
            )
        if self._echo_reference and self._echo_reference.device_gain == 4:
            # Firmware multiplies by four before its DAC. Keep that operation
            # linear: an immediate peak attack and 200 ms release preserve quiet
            # samples without allowing loud speech/music to clip in the device.
            peak = float(np.max(np.abs(audio)))
            required = min(1.0, (32767 / 32768 / 4) / max(peak, 1e-9))
            self._speaker_gain = min(required, self._speaker_gain + self.BLOCK_SEC / 0.2)
            audio = audio * self._speaker_gain
        try:
            if opened:
                if self._stream.write(audio.reshape(-1, 1)) is True:
                    self._output_underflows += 1
                    if self._output_underflows == 1:
                        print("[Playback Service] 混音输出缓冲欠载", flush=True)
                if self._echo_reference:
                    self._echo_reference.send(audio, latency)
            else:
                self._stopped.wait(self.BLOCK_SEC)
        finally:
            # Even an unavailable speaker must release capture for this turn.
            for token in completed:
                if isinstance(token, tuple) and token[0] == "prompt":
                    category, request_id = token[1], token[2]
                    if category == "wake" and self.on_wake_complete:
                        self.on_wake_complete(request_id)
                    elif category == "system" and self.on_system_complete:
                        self.on_system_complete(request_id)
                elif self.on_turn_complete:
                    self.on_turn_complete()

    def _close_stream(self, drain=False):
        had_stream = self._stream is not None
        try:
            super()._close_stream(drain=drain)
        finally:
            if had_stream and self._echo_reference:
                self._echo_reference.end()

    def close(self):
        self._stopped.set()
        self._ready.set()
        self._worker.join(timeout=2.0)
        if self._echo_reference and not self._worker.is_alive():
            self._echo_reference.close()
