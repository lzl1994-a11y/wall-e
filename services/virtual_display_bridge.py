"""Bridge a generic Xvfb application to the existing chest-TFT stream."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

from services.tft_preview_server import TftPreviewServer
from services.virtual_display import MjpegFrameSource, VirtualDisplay, VirtualDisplaySettings


class VirtualDisplayTftBridge:
    """Own a virtual GUI session while delegating TFT transport to its owner.

    The bridge never opens a TCP listener: ``TftPreviewServer`` remains the
    single owner of the ESP32 connection.  It also deliberately has no ROS or
    FCEUX imports, so it can serve a game, a launcher, or another GUI program.
    """

    def __init__(
        self,
        tft_server: TftPreviewServer,
        *,
        settings: VirtualDisplaySettings = VirtualDisplaySettings(),
        display: VirtualDisplay | None = None,
        frames: MjpegFrameSource | None = None,
        on_stream_lost: Callable[[], None] | None = None,
    ) -> None:
        self._tft_server = tft_server
        self._settings = settings
        self._display = display or VirtualDisplay(settings)
        self._frames = frames or MjpegFrameSource(settings)
        self._on_stream_lost = on_stream_lost
        # The stream type is intentionally duck-typed.  The legacy TFT server
        # remains untouched; a game-capable host may provide ``open_jpeg_stream``
        # as a separate adapter without making that method part of the legacy API.
        self._stream: Any = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self, command: Sequence[str]) -> None:
        """Start only after an exclusive connected TFT stream is available."""
        if self.running:
            raise RuntimeError("virtual-display bridge is already running")
        self._display.start()
        self._frames.start()
        open_stream = getattr(self._tft_server, "open_jpeg_stream", None)
        if not callable(open_stream):
            self._frames.stop()
            self._display.stop()
            raise RuntimeError("chest TFT game stream adapter is unavailable")
        stream = open_stream(fps=self._settings.fps)
        if stream is None:
            self._frames.stop()
            self._display.stop()
            raise RuntimeError("chest TFT is unavailable or busy")
        self._stream = stream
        try:
            self._display.launch(command)
        except Exception:
            stream.close()
            self._stream = None
            self._frames.stop()
            self._display.stop()
            raise
        self._stop.clear()
        self._worker = threading.Thread(target=self._forward_frames, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self._worker = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()
        self._frames.stop()
        self._display.stop()

    def _forward_frames(self) -> None:
        sequence = -1
        period = 1.0 / max(1, self._settings.fps)
        while not self._stop.is_set():
            current_sequence, jpeg = self._frames.latest
            stream = self._stream
            if stream is None:
                return
            if jpeg is not None and current_sequence != sequence:
                sequence = current_sequence
                if not stream.send_jpeg(jpeg):
                    if self._on_stream_lost is not None:
                        self._on_stream_lost()
                    return
            self._stop.wait(period)


__all__ = ["VirtualDisplayTftBridge"]
