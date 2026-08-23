"""Mirror the active tracking camera to the chest TFT without owning a camera."""

from __future__ import annotations

import threading
import time
from typing import Any

from services.vision_pipeline_protocol import VISION_PIPELINE_START


class TrackingTftPreview:
    """Run one preemptible TFT stream while the BPU tracking pipeline is active.

    The stream consumes ``/camera_frame`` through the normal frame provider.  Its
    lease tells camera_capture_node to relay the tracking pipeline's ``/image``;
    it never opens a second V4L2 camera.
    """

    STREAM_DURATION_MS = 60_000

    def __init__(self, server: Any, frame_provider: Any, *, fps: int, logger: Any = None) -> None:
        self._server = server
        self._frame_provider = frame_provider
        self._fps = int(fps)
        self._logger = logger
        self._enabled = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_command(self, command: str) -> None:
        self.set_enabled(command == VISION_PIPELINE_START)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._stop_event.set()
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="tracking-tft-preview",
                daemon=True,
            )
            self._thread.start()

    def pause(self) -> bool:
        """Stop the active stream so a one-shot photo/inspection can use TFT."""
        with self._lock:
            was_enabled = self._enabled
            if was_enabled:
                self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        return was_enabled

    def resume(self) -> None:
        with self._lock:
            if not self._enabled:
                return
            thread = self._thread
        if thread is not None and thread.is_alive():
            # ``pause`` normally joins quickly.  Keep the handoff correct even
            # if a blocked network write needs longer than that grace period.
            threading.Thread(
                target=self._resume_after,
                args=(thread,),
                name="tracking-tft-preview-resume",
                daemon=True,
            ).start()
            return
        self.set_enabled(True)

    def _resume_after(self, thread: threading.Thread) -> None:
        thread.join()
        self.set_enabled(True)

    def stop(self) -> None:
        self.set_enabled(False)
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            result = self._server.send_camera_preview(
                self._frame_provider,
                duration_ms=self.STREAM_DURATION_MS,
                hold_ms=0,
                fps=self._fps,
                should_stop=self._stop_event.is_set,
            )
            if self._stop_event.is_set():
                return
            # A photo/inspection may have won the TFT stream lock just before
            # pause() was called. Retry calmly once that short stream finishes.
            if getattr(result, "busy", False):
                self._stop_event.wait(0.2)
            elif not getattr(result, "last_frame", None):
                self._stop_event.wait(0.5)
