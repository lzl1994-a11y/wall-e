"""Linux ALSA capture without PortAudio's global timer/device probing."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from typing import Callable

import numpy as np


def native_capture_available() -> bool:
    """Return whether the resilient Linux capture backend can be used."""

    return os.name == "posix" and shutil.which("arecord") is not None


class ArecordInputStream:
    """Small ``sounddevice.InputStream`` compatible wrapper around arecord.

    PortAudio enumerates every ALSA PCM and opens ``/dev/snd/timer``.  On the
    deployed 4.14 RDK kernel, unplugging a USB UAC device while that timer is
    being released can leave every later audio process blocked in D state.
    arecord talks to the selected PCM directly and avoids that global timer.
    """

    def __init__(
        self,
        *,
        device: str,
        channels: int,
        samplerate: int,
        blocksize: int,
        callback: Callable,
    ):
        self.device = device
        self.channels = int(channels)
        self.samplerate = int(samplerate)
        self.blocksize = int(blocksize)
        self.callback = callback
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def active(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None and not self._stop_event.is_set()

    def start(self) -> None:
        if self.active:
            return
        command = [
            "arecord",
            "-q",
            "-D",
            self.device,
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(self.samplerate),
            "-c",
            str(self.channels),
        ]
        self._stop_event.clear()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._read_loop,
            name="alsa-capture-reader",
            daemon=True,
        )
        self._reader.start()

    def stop(self) -> None:
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self) -> None:
        self.stop()
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=0.5)
        self._process = None
        self._reader = None

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        frame_bytes = self.blocksize * self.channels * 2
        pending = bytearray()
        while not self._stop_event.is_set():
            chunk = process.stdout.read(frame_bytes - len(pending))
            if not chunk:
                break
            pending.extend(chunk)
            if len(pending) < frame_bytes:
                continue
            pcm = np.frombuffer(bytes(pending), dtype=np.int16).reshape(
                self.blocksize, self.channels
            )
            audio = pcm.astype(np.float32) / 32768.0
            pending.clear()
            self.callback(audio, self.blocksize, None, None)
