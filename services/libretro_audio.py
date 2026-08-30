"""Low-latency, output-only PCM sink used by the isolated libretro probe.

This deliberately does not use the robot's voice pipeline: it opens one
sounddevice *output* stream and accepts libretro's interleaved stereo PCM.
"""

from __future__ import annotations

import ctypes
import threading

from services.usb_devices import DEFAULT_CONFIG_PATH, resolve_audio_device


class LibretroAudioPlayer:
    """Bounded stereo PCM buffer backed by a single output-only stream."""

    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        device: int | None = None,
        config_path=DEFAULT_CONFIG_PATH,
        sounddevice_module=None,
        max_buffer_ms: int = 120,
    ) -> None:
        if sounddevice_module is None:
            import sounddevice as sounddevice_module

        self._sd = sounddevice_module
        self.sample_rate = sample_rate
        self.device = device if device is not None else self._resolve_device(config_path)
        self._limit = max(1, sample_rate * max_buffer_ms // 1_000) * 4
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream = None

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
        if self._stream is not None:
            return
        self._stream = self._sd.RawOutputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=2,
            dtype="int16",
            latency="low",
            callback=self._output_callback,
        )
        self._stream.start()

    def push_batch(self, samples: ctypes.POINTER(ctypes.c_short), frames: int) -> None:
        if not samples or frames <= 0:
            return
        payload = ctypes.string_at(samples, frames * 4)
        with self._lock:
            self._buffer.extend(payload)
            overflow = len(self._buffer) - self._limit
            if overflow > 0:
                del self._buffer[:overflow]

    def push_sample(self, left: int, right: int) -> None:
        frame = (ctypes.c_short * 2)(left, right)
        self.push_batch(frame, 1)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort()
        finally:
            stream.close()

    def _output_callback(self, outdata, frames, _time_info, _status) -> None:
        wanted = frames * 4
        with self._lock:
            payload = bytes(self._buffer[:wanted])
            del self._buffer[: len(payload)]
        outdata[: len(payload)] = payload
        if len(payload) < wanted:
            outdata[len(payload) : wanted] = b"\x00" * (wanted - len(payload))


__all__ = ["LibretroAudioPlayer"]
