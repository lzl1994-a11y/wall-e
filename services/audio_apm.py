"""Persistent native GStreamer/WebRTC APM bridge for the microphone pipeline."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable


class WebRTCApm:
    """Feed native microphone PCM to one long-lived WebRTC APM process."""

    OUTPUT_RATE = 16000

    def __init__(self, on_pcm: Callable[[bytes], None], *, pre_gain_db: float = 6.0):
        self._on_pcm = on_pcm
        self.pre_gain_db = float(min(24.0, max(-12.0, pre_gain_db)))
        self._running = False
        self._input_rate = 0
        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()

    @staticmethod
    def available() -> bool:
        return shutil.which("gst-launch-1.0") is not None

    def start(self, input_rate: int) -> bool:
        with self._lock:
            if self._running and self._input_rate == input_rate:
                return True
            self.stop()
            if not self.available():
                print("[AudioPipeline] GStreamer 未安装，回退未增强采集")
                return False
            try:
                self._process = subprocess.Popen(
                    self._command(input_rate), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0,
                )
            except OSError as exc:
                print(f"[AudioPipeline] GStreamer APM 启动失败: {exc}")
                return False
            self._running, self._input_rate = True, input_rate
            self._queue = queue.Queue(maxsize=100)
            self._writer = threading.Thread(target=self._write, name="wali-apm-input", daemon=True)
            self._reader = threading.Thread(target=self._read, name="wali-apm-output", daemon=True)
            self._writer.start(); self._reader.start()
        print(f"[AudioPipeline] WebRTC APM 常驻: {input_rate}Hz -> 16000Hz, pre-gain={self.pre_gain_db:g}dB")
        return True

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            running, self._running = self._running, False
            if running:
                try: self._queue.put_nowait(None)
                except queue.Full: pass
        if process:
            for stream in (process.stdin, process.stdout):
                try: stream.close()
                except Exception: pass
            try: process.terminate(); process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try: process.kill()
                except OSError: pass
        for thread in (self._writer, self._reader):
            if thread and thread is not threading.current_thread(): thread.join(timeout=1)
        self._writer = self._reader = None
        self._input_rate = 0

    def submit(self, pcm: bytes) -> bool:
        if not self._running: return False
        try:
            self._queue.put_nowait(pcm)
            return True
        except queue.Full:
            print("[AudioPipeline] APM 输入积压，丢弃一帧音频")
            return False

    def _command(self, input_rate: int) -> list[str]:
        gain = 10 ** (self.pre_gain_db / 20.0)
        return [
            "gst-launch-1.0", "-q", "fdsrc", "fd=0", "blocksize=960", "!",
            "rawaudioparse", "format=pcm", "pcm-format=s16le", f"sample-rate={input_rate}", "num-channels=1", "!",
            "audioconvert", "!", "audioresample", "!",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1", "!",
            "volume", f"volume={gain:.9f}", "!",
            "webrtcdsp", "echo-cancel=false", "high-pass-filter=true", "noise-suppression=true",
            "noise-suppression-level=moderate", "gain-control=true", "gain-control-mode=fixed-digital",
            "target-level-dbfs=3", "limiter=true", "!", "fdsink", "fd=1", "sync=false",
        ]

    def _write(self) -> None:
        while self._running:
            frame = self._queue.get()
            if frame is None: return
            try:
                if not self._process or not self._process.stdin: return
                self._process.stdin.write(frame)
            except (BrokenPipeError, OSError):
                print("[AudioPipeline] GStreamer APM 输入管道已关闭")
                return

    def _read(self) -> None:
        pending = bytearray()
        frame_bytes = self.OUTPUT_RATE * 30 // 1000 * 2
        try:
            if not self._process or not self._process.stdout: return
            while self._running:
                chunk = os.read(self._process.stdout.fileno(), 4096)
                if not chunk: return
                pending.extend(chunk)
                while len(pending) >= frame_bytes:
                    frame = bytes(pending[:frame_bytes]); del pending[:frame_bytes]
                    if self._running: self._on_pcm(frame)
        except OSError:
            if self._running: print("[AudioPipeline] GStreamer APM 输出管道已关闭")
