"""Persistent native GStreamer/WebRTC APM bridge for the microphone pipeline."""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable


class WebRTCApm:
    """Feed native microphone PCM to one long-lived WebRTC APM process."""

    MAX_PENDING_FRAMES = 4
    MAX_OUTPUT_STALL_SEC = 0.25

    def __init__(self, on_pcm: Callable[[bytes], None], *, pre_gain_db: float = 6.0,
                 output_rate: int = 16000, frame_ms: int = 30, echo_cancel: bool = False,
                 capture_delay_ms: int = 20):
        if output_rate not in (16000, 48000) or frame_ms not in (10, 20, 30):
            raise ValueError("unsupported APM PCM format")
        self.output_rate = output_rate
        self.frame_ms = frame_ms
        self.echo_cancel = echo_cancel
        if not 0 <= capture_delay_ms <= 500:
            raise ValueError("invalid capture delay")
        self.capture_delay_ms = capture_delay_ms
        self._native = None
        self._reference = None
        self._on_pcm = on_pcm
        self.pre_gain_db = float(min(24.0, max(-12.0, pre_gain_db)))
        self._running = False
        self._input_rate = 0
        self._process: subprocess.Popen | None = None
        self._queue: queue.Queue[bytes | tuple[bytes, float] | None] = queue.Queue(maxsize=self.MAX_PENDING_FRAMES)
        self._writer: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self._overloaded = threading.Event()
        self._first_input_at: float | None = None
        self._last_input_at: float | None = None
        self._last_output_at: float | None = None
        self.dropped_frames = 0

    @staticmethod
    def available() -> bool:
        return shutil.which("gst-launch-1.0") is not None

    def start(self, input_rate: int) -> bool:
        with self._lock:
            if self._running and self._input_rate == input_rate:
                return True
            self.stop()
            if self.echo_cancel:
                if (input_rate, self.output_rate, self.frame_ms) != (48000, 48000, 20):
                    raise ValueError("intercom AEC requires 48 kHz mono / 20 ms")
                from services.audio.echo_reference import EchoReferenceReceiver
                from services.audio.native_apm import NativeEchoApm
                try:
                    self._native = NativeEchoApm(self.pre_gain_db)
                    self._reference = EchoReferenceReceiver()
                except (OSError, RuntimeError) as exc:
                    if self._native:
                        self._native.close()
                        self._native = None
                    print(f"[AudioPipeline] AEC 启动失败，采集没有回声消除: {exc}", flush=True)
                    return False
            elif not self.available():
                print("[AudioPipeline] GStreamer 未安装，回退未增强采集")
                return False
            else:
                try:
                    self._process = subprocess.Popen(
                        self._command(input_rate), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, bufsize=0,
                    )
                except OSError as exc:
                    print(f"[AudioPipeline] GStreamer APM 启动失败: {exc}")
                    return False
            self._running, self._input_rate = True, input_rate
            self._queue = queue.Queue(maxsize=self.MAX_PENDING_FRAMES)
            self._overloaded.clear()
            self._first_input_at = self._last_input_at = self._last_output_at = None
            self.dropped_frames = 0
            self._writer = threading.Thread(target=self._write_native if self.echo_cancel else self._write,
                                            name="wali-apm-input", daemon=True)
            self._writer.start()
            if not self.echo_cancel:
                self._reader = threading.Thread(target=self._read, name="wali-apm-output", daemon=True)
                self._reader.start()
        print(f"[AudioPipeline] WebRTC APM 常驻: {input_rate}Hz -> {self.output_rate}Hz, "
              f"pre-gain={self.pre_gain_db:g}dB, AEC={'ON (native, device latency)' if self.echo_cancel else 'OFF'}", flush=True)
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
        self._native = self._reference = None
        self._input_rate = 0

    @property
    def overloaded(self) -> bool:
        """Whether the native processor stopped consuming live audio."""
        return self._overloaded.is_set()

    def submit(self, pcm: bytes) -> bool:
        if not self._running or self.overloaded: return False
        now = time.monotonic()
        # An input gap is not evidence that the native processor stalled.
        # If it produced output after the previous input, start a new budget
        # for this frame instead of timing the source's idle interval.
        caught_up = (self._last_input_at is None or (
            self._last_output_at is not None and self._last_output_at >= self._last_input_at
        ))
        if caught_up:
            self._first_input_at = now
        last_progress = max(self._first_input_at,
                            self._last_output_at if self._last_output_at is not None else self._first_input_at)
        if now - last_progress > self.MAX_OUTPUT_STALL_SEC:
            self._overloaded.set()
            print("[AudioPipeline] APM 输出停滞超过实时预算，切换到直采样回退")
            return False
        self._last_input_at = now
        try:
            self._queue.put_nowait((pcm, now) if self.echo_cancel else pcm)
            return True
        except queue.Full:
            # A capture burst is not a wedged processor. Keep only current
            # frames; failure is determined by OUTPUT progress, not fullness.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped_frames += 1
            if self.dropped_frames == 1:
                print("[AudioPipeline] APM 突发输入，丢弃旧帧以保持实时")
            if not self._running:
                return False
            self._queue.put_nowait((pcm, now) if self.echo_cancel else pcm)
            return True

    def _command(self, input_rate: int) -> list[str]:
        gain = 10 ** (self.pre_gain_db / 20.0)
        return [
            "gst-launch-1.0", "-q", "fdsrc", "fd=0", f"blocksize={input_rate * self.frame_ms // 1000 * 2}", "!",
            "rawaudioparse", "format=pcm", "pcm-format=s16le", f"sample-rate={input_rate}", "num-channels=1", "!",
            "audioconvert", "!", "audioresample", "!",
            f"audio/x-raw,format=S16LE,layout=interleaved,rate={self.output_rate},channels=1", "!",
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

    def _write_native(self) -> None:
        native, reference = self._native, self._reference
        last_render_at = None
        delay_ms = 0
        active = False
        silence = bytes(1920)
        processed = 0
        last_report = time.monotonic()
        try:
            while self._running:
                pcm = self._queue.get()
                if pcm is None:
                    return
                queued_at = time.monotonic()
                if isinstance(pcm, tuple):
                    pcm, queued_at = pcm
                for render, delay, sent_at in reference.drain():
                    if render:
                        native.render(render)
                        last_render_at, delay_ms, active = sent_at, delay, True
                    else:
                        active = False
                # A closed/idle speaker renders silence. Do not insert silence
                # between output writes: the sound card may still be draining.
                if (not active or last_render_at is None or
                        time.monotonic() - last_render_at > delay_ms / 1000 + 0.04):
                    native.render(silence)
                queue_delay = max(0, round((time.monotonic() - queued_at) * 1000))
                result = native.capture(pcm, min(500, delay_ms + self.capture_delay_ms + queue_delay))
                processed += 1
                if self._running and not self.overloaded:
                    self._last_output_at = time.monotonic()
                    self._on_pcm(result)
                if time.monotonic() - last_report >= 10:
                    print(f"[AudioPipeline] AEC 运行统计: capture_frames={processed} "
                          f"reference_frames={reference.frames} reference_drops={reference.dropped} "
                          f"queue_drops={self.dropped_frames} metrics={native.metrics()}", flush=True)
                    last_report = time.monotonic()
        except Exception as exc:
            self._overloaded.set()
            print(f"[AudioPipeline] AEC 处理失败，回声消除已失效: {exc}", flush=True)
        finally:
            print(f"[AudioPipeline] AEC 参考统计: frames={reference.frames} dropped={reference.dropped} "
                  f"metrics={native.metrics()}", flush=True)
            reference.close()
            native.close()

    def _read(self) -> None:
        pending = bytearray()
        frame_bytes = self.output_rate * self.frame_ms // 1000 * 2
        try:
            if not self._process or not self._process.stdout: return
            while self._running:
                chunk = os.read(self._process.stdout.fileno(), 4096)
                if not chunk: return
                pending.extend(chunk)
                while len(pending) >= frame_bytes:
                    frame = bytes(pending[:frame_bytes]); del pending[:frame_bytes]
                    if self._running and not self.overloaded:
                        self._last_output_at = time.monotonic()
                        self._on_pcm(frame)
        except OSError:
            if self._running: print("[AudioPipeline] GStreamer APM 输出管道已关闭")
