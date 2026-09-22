"""Pace PCM publication so a ROS topic never receives an audio burst."""

from __future__ import annotations

import math
import queue
import threading
import time

import numpy as np


class PacedPCMOutput:
    """Queue PCM locally and emit it at its audible rate.

    A short initial lead lets the receiver prebuffer audio.  Subsequent chunks
    are scheduled on the PCM clock, preventing a fast TTS decoder from
    overflowing a shallow transport queue.
    """

    _STOP = object()

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_ms: int,
        prebuffer_ms: float,
        max_queued_audio_sec: float,
        on_pcm,
        on_turn_end,
    ):
        self.sample_rate = max(1, int(sample_rate))
        self.chunk_frames = max(1, round(self.sample_rate * max(20, int(chunk_ms)) / 1000))
        self.prebuffer_frames = max(
            0, round(self.sample_rate * max(0.0, float(prebuffer_ms)) / 1000)
        )
        queue_blocks = max(
            1,
            math.ceil(max(1.0, float(max_queued_audio_sec)) * self.sample_rate / self.chunk_frames),
        )
        self._queue = queue.Queue(maxsize=queue_blocks)
        self._on_pcm = on_pcm
        self._on_turn_end = on_turn_end
        self._stopped = threading.Event()
        self._clock_started_at = None
        self._frames_sent = 0
        self._worker = threading.Thread(
            target=self._run, name="tts-pcm-pacer", daemon=True
        )
        self._worker.start()

    def submit(self, samples: np.ndarray) -> None:
        audio = np.asarray(samples, dtype=np.int16).reshape(-1)
        for start in range(0, audio.size, self.chunk_frames):
            if self._stopped.is_set():
                return
            # Copy because full-sentence buffers are reused by their producer.
            self._queue.put(("pcm", audio[start : start + self.chunk_frames].copy()))

    def finish_turn(self, turn_id: str) -> None:
        if not self._stopped.is_set():
            self._queue.put(("turn_end", turn_id))

    def close(self) -> None:
        self._stopped.set()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if item is self._STOP:
                    return
                item_type, value = item
                if item_type == "pcm":
                    self._wait_for_pcm_slot(value.size)
                    if not self._stopped.is_set():
                        self._on_pcm(value)
                        self._frames_sent += value.size
                else:
                    self._on_turn_end(value)
                    self._clock_started_at = None
                    self._frames_sent = 0
            finally:
                self._queue.task_done()

    def _wait_for_pcm_slot(self, frame_count: int) -> None:
        if self._clock_started_at is None:
            self._clock_started_at = time.monotonic()
        due_at = self._clock_started_at + max(
            0.0, (self._frames_sent - self.prebuffer_frames) / self.sample_rate
        )
        self._stopped.wait(max(0.0, due_at - time.monotonic()))

