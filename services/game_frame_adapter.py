"""Asynchronously route libretro video frames to the existing TFT stream."""

from __future__ import annotations

import threading
import time

import numpy as np


class GameFrameAdapter:
    """Keep only the newest raw frame so encoding never blocks the game core."""

    def __init__(self, stream, *, fps: float = 10.0) -> None:
        self._stream = stream
        self._period = 1.0 / max(1.0, float(fps))
        self._condition = threading.Condition()
        self._latest: tuple[bytes, int, int, int] | None = None
        self._closed = False
        self.callbacks = 0
        self.overwritten = 0
        self.encoded = 0
        self._worker = threading.Thread(
            target=self._encode_worker, name="game-tft-encoder", daemon=True
        )
        self._worker.start()

    def submit_frame(self, raw: bytes, width: int, height: int, pitch: int) -> None:
        """Store a frame without resizing or JPEG encoding on the caller thread."""
        if not raw or width <= 0 or height <= 0 or pitch < width * 4:
            return
        if len(raw) < pitch * height:
            return
        frame = (raw if isinstance(raw, bytes) else bytes(raw), width, height, pitch)
        with self._condition:
            if self._closed:
                return
            self.callbacks += 1
            if self._latest is not None:
                self.overwritten += 1
            self._latest = frame
            self._condition.notify()

    def close(self) -> None:
        """Discard any unencoded frame and wait for an active encode to finish."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            if self._latest is not None:
                self.overwritten += 1
                self._latest = None
            self._condition.notify_all()
        self._worker.join()

    def _encode_worker(self) -> None:
        next_frame_at = 0.0
        while True:
            with self._condition:
                while not self._closed and self._latest is None:
                    self._condition.wait()
                if self._closed:
                    return
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    continue
                raw, width, height, pitch = self._latest
                self._latest = None

            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
            self._stream.send_bgr(image[:, :width, :3])
            self.encoded += 1
            next_frame_at = time.monotonic() + self._period


__all__ = ["GameFrameAdapter"]
