"""Output-only PCM worker used by the isolated libretro probe.

The FC core only copies PCM into a bounded queue. A dedicated worker owns the
PortAudio stream and performs the potentially blocking USB writes, so video
encoding cannot make the real-time output callback run out of samples.
"""

from __future__ import annotations

from collections import deque
import ctypes
import threading

from services.usb_devices import DEFAULT_CONFIG_PATH, resolve_audio_device


class LibretroAudioPlayer:
    """Independent output worker with a bounded stereo PCM ring buffer."""

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        device: int | None = None,
        config_path=DEFAULT_CONFIG_PATH,
        sounddevice_module=None,
        max_buffer_ms: int = 200,
        prebuffer_ms: int = 60,
        latency: float = 0.06,
    ) -> None:
        if sounddevice_module is None:
            import sounddevice as sounddevice_module

        self._sd = sounddevice_module
        self.sample_rate = sample_rate
        self.device = device if device is not None else self._resolve_device(config_path)
        self._limit = max(1, sample_rate * max_buffer_ms // 1_000) * 4
        self._prebuffer = min(
            self._limit, max(1, sample_rate * prebuffer_ms // 1_000) * 4
        )
        self._latency = latency
        self._chunks: deque[bytes] = deque()
        self._queued_bytes = 0
        self._condition = threading.Condition()
        self._stopping = False
        self._stream = None
        self._worker: threading.Thread | None = None
        self.error: Exception | None = None

    def _resolve_device(self, config_path):
        resolution = resolve_audio_device(
            "output", config_path, sounddevice_module=self._sd
        )
        if resolution.configured:
            if not resolution.available:
                raise RuntimeError("configured voice audio device is unavailable")
            return resolution.index
        devices = self._sd.query_devices()
        for index, candidate in enumerate(devices):
            if int(candidate.get("max_output_channels", 0)) >= 2:
                return index
        raise RuntimeError("no stereo output audio device found")

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._play_worker, daemon=True)
        self._worker.start()

    def push_batch(self, samples: ctypes.POINTER(ctypes.c_short), frames: int) -> None:
        if not samples or frames <= 0:
            return
        self._push(ctypes.string_at(samples, frames * 4))

    def push_sample(self, left: int, right: int) -> None:
        frame = (ctypes.c_short * 2)(left, right)
        self.push_batch(frame, 1)

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2.0)
        self._worker = None

    def _push(self, payload: bytes) -> None:
        with self._condition:
            if self._stopping:
                return
            self._chunks.append(payload)
            self._queued_bytes += len(payload)
            while self._queued_bytes > self._limit and self._chunks:
                self._queued_bytes -= len(self._chunks.popleft())
            self._condition.notify()

    def _take(self, *, require_prebuffer: bool) -> bytes | None:
        with self._condition:
            while not self._stopping and (
                not self._chunks or (require_prebuffer and self._queued_bytes < self._prebuffer)
            ):
                self._condition.wait(timeout=0.1)
            if self._stopping:
                return None
            payload = self._chunks.popleft()
            self._queued_bytes -= len(payload)
            return payload

    def _play_worker(self) -> None:
        stream = None
        try:
            first = self._take(require_prebuffer=True)
            if first is None:
                return
            stream = self._sd.RawOutputStream(
                samplerate=self.sample_rate,
                device=self.device,
                channels=2,
                dtype="int16",
                latency=self._latency,
            )
            self._stream = stream
            stream.start()
            stream.write(first)
            while True:
                payload = self._take(require_prebuffer=False)
                if payload is None:
                    return
                stream.write(payload)
        except Exception as exc:
            self.error = exc
        finally:
            self._stream = None
            if stream is not None:
                try:
                    stream.abort()
                finally:
                    stream.close()


__all__ = ["LibretroAudioPlayer"]
