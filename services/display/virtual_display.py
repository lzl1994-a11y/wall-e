"""Generic virtual-display lifecycle and latest-frame capture.

The module deliberately has no ROS, TFT, or emulator dependency.  Any GUI
application can render to :class:`VirtualDisplay`; :class:`MjpegFrameSource`
then exposes the latest desktop frame for a downstream display adapter.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class VirtualDisplaySettings:
    display_number: int = 91
    width: int = 768
    height: int = 720
    fps: int = 30
    startup_timeout_sec: float = 5.0

    @property
    def display(self) -> str:
        return f":{self.display_number}"

    @property
    def x11_socket(self) -> Path:
        return Path(f"/tmp/.X11-unix/X{self.display_number}")


class MjpegParser:
    """Extract complete JPEGs from an arbitrary byte stream."""

    SOI = b"\xff\xd8"
    EOI = b"\xff\xd9"

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        frames: list[bytes] = []
        while True:
            start = self._buffer.find(self.SOI)
            if start < 0:
                # Preserve a trailing 0xff: it may be the first byte of SOI.
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                break
            if start:
                del self._buffer[:start]
            end = self._buffer.find(self.EOI, len(self.SOI))
            if end < 0:
                break
            end += len(self.EOI)
            frames.append(bytes(self._buffer[:end]))
            del self._buffer[:end]
        return frames


class VirtualDisplay:
    """Own one Xvfb instance and optionally launch a GUI application on it."""

    def __init__(self, settings: VirtualDisplaySettings = VirtualDisplaySettings()) -> None:
        self.settings = settings
        self._xvfb: subprocess.Popen | None = None
        self._application: subprocess.Popen | None = None

    def start(self) -> None:
        if self._xvfb is not None and self._xvfb.poll() is None:
            return
        self._xvfb = subprocess.Popen(
            [
                "Xvfb",
                self.settings.display,
                "-screen",
                "0",
                f"{self.settings.width}x{self.settings.height}x24",
                "+extension",
                "GLX",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.settings.startup_timeout_sec
        while time.monotonic() < deadline:
            if self._xvfb.poll() is not None:
                raise RuntimeError("Xvfb exited during startup")
            if self.settings.x11_socket.exists():
                return
            time.sleep(0.05)
        self.stop()
        raise TimeoutError(f"Xvfb did not create {self.settings.display}")

    def launch(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.start()
        if self._application is not None and self._application.poll() is None:
            raise RuntimeError("a virtual-display application is already running")
        env = os.environ.copy()
        env.update({"DISPLAY": self.settings.display, "LIBGL_ALWAYS_SOFTWARE": "1"})
        if environment:
            env.update(environment)
        self._application = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    def stop_application(self) -> None:
        self._stop_process(self._application)
        self._application = None

    def stop(self) -> None:
        self.stop_application()
        self._stop_process(self._xvfb)
        self._xvfb = None

    @staticmethod
    def _stop_process(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


class MjpegFrameSource:
    """Capture a virtual X11 display and retain only its newest JPEG frame."""

    def __init__(self, settings: VirtualDisplaySettings) -> None:
        self.settings = settings
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._sequence = 0

    @property
    def latest(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._sequence, self._latest_frame

    @staticmethod
    def command_for(settings: VirtualDisplaySettings) -> list[str]:
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "x11grab",
            "-framerate", str(settings.fps),
            "-video_size", f"{settings.width}x{settings.height}",
            "-i", f"{settings.display}.0",
            "-an",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "-f", "image2pipe",
            "pipe:1",
        ]

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._stop.clear()
        self._process = subprocess.Popen(
            self.command_for(self.settings),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        parser = MjpegParser()
        while not self._stop.is_set():
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                return
            frames = parser.feed(chunk)
            if not frames:
                continue
            with self._lock:
                self._latest_frame = frames[-1]
                self._sequence += len(frames)


__all__ = ["MjpegFrameSource", "MjpegParser", "VirtualDisplay", "VirtualDisplaySettings"]
