"""Bridge a generic Xvfb application to the existing chest-TFT stream."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

from services.tft_preview_server import TftPreviewServer
from services.virtual_display import MjpegFrameSource, VirtualDisplay, VirtualDisplaySettings


class _LegacyFrameProvider:
    """Adapt the newest virtual-display frame to the legacy preview API."""

    def __init__(self, frames: MjpegFrameSource, stop: threading.Event) -> None:
        self._frames = frames
        self._stop = stop

    def capture_stream(
        self,
        *,
        duration_ms: int,
        fps: int,
        on_frame: Callable[[bytes, int], None] | None = None,
        on_source_frame: Callable[[bytes, int], None] | None = None,
        on_no_new_frame: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        **_: object,
    ) -> bytes | None:
        callback = on_frame or on_source_frame
        deadline = time.monotonic() + max(0.1, duration_ms / 1000.0)
        period = 1.0 / max(1, int(fps))
        last_sequence = -1
        latest_frame: bytes | None = None

        while time.monotonic() < deadline:
            if self._stop.is_set() or (should_stop is not None and should_stop()):
                break
            sequence, jpeg = self._frames.latest
            if jpeg is not None and sequence != last_sequence:
                last_sequence = sequence
                latest_frame = bytes(jpeg)
                if callback is not None:
                    callback(latest_frame, sequence)
            elif on_no_new_frame is not None:
                on_no_new_frame()
            time.sleep(min(period, max(0.0, deadline - time.monotonic())))
        return latest_frame


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
        self._legacy_preview = False
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
        if callable(open_stream):
            stream = open_stream(fps=self._settings.fps)
            if stream is None:
                self._frames.stop()
                self._display.stop()
                raise RuntimeError("chest TFT is unavailable or busy")
            self._stream = stream
        else:
            # Keep the legacy TftPreviewServer API untouched.  This fallback
            # sends short preview windows through its existing public method.
            send_preview = getattr(self._tft_server, "send_camera_preview", None)
            connected = getattr(self._tft_server, "device_connected", True)
            if not callable(send_preview) or not bool(connected):
                self._frames.stop()
                self._display.stop()
                raise RuntimeError("chest TFT game stream adapter is unavailable")
            self._legacy_preview = True
        try:
            self._display.launch(command)
        except Exception:
            if self._stream is not None:
                self._stream.close()
            self._stream = None
            self._legacy_preview = False
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
        self._legacy_preview = False
        self._frames.stop()
        self._display.stop()

    def _forward_frames(self) -> None:
        if self._legacy_preview:
            self._forward_legacy_preview()
            return
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

    def _forward_legacy_preview(self) -> None:
        send_preview = self._tft_server.send_camera_preview
        provider = _LegacyFrameProvider(self._frames, self._stop)
        while not self._stop.is_set():
            result = send_preview(
                provider,
                duration_ms=max(500, int(1000 / max(1, self._settings.fps))),
                hold_ms=0,
                fps=self._settings.fps,
                should_stop=self._stop.is_set,
            )
            if self._stop.is_set():
                return
            if getattr(result, "error", "") and self._on_stream_lost is not None:
                self._on_stream_lost()
                return


__all__ = ["VirtualDisplayTftBridge"]
