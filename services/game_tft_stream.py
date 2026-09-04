"""Game-frame adaptation built on the shared persistent TFT transport."""

from __future__ import annotations

import time

from services.tft_preview_server import PersistentTftStream, TftPreviewServer


def prepare_game_jpeg(jpeg: bytes, *, quality: int = 70) -> bytes | None:
    """Fit an upright game frame to the TFT without the camera's rotation."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    raw = bytes(jpeg or b"")
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        return None
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    scale = min(240.0 / width, 240.0 / height)
    target = (
        min(240, max(1, int(width * scale + 0.5))),
        min(240, max(1, int(height * scale + 0.5))),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, target, interpolation=interpolation)
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), min(100, max(1, quality))]
    )
    return encoded.tobytes() if ok else None


def prepare_game_bgr(image, *, quality: int = 70) -> bytes | None:
    """Fit a raw upright BGR frame to the TFT with one JPEG encode."""
    try:
        import cv2
    except ImportError:
        return None
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    scale = min(240.0 / width, 240.0 / height)
    target = (
        min(240, max(1, int(width * scale + 0.5))),
        min(240, max(1, int(height * scale + 0.5))),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = image if target == (width, height) else cv2.resize(
        image, target, interpolation=interpolation
    )
    ok, encoded = cv2.imencode(
        ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), min(100, max(1, quality))]
    )
    return encoded.tobytes() if ok else None


class GameTftStream:
    def __init__(
        self,
        server: "GameTftStreamServer",
        transport: PersistentTftStream,
    ) -> None:
        self._server = server
        self._transport = transport
        self._closed = False
        self.prepare_seconds = 0.0
        self.send_seconds = 0.0
        self.prepare_attempts = 0

    def send_jpeg(self, jpeg: bytes) -> bool:
        if self._closed:
            return False
        started = time.perf_counter()
        frame = prepare_game_jpeg(jpeg, quality=self._server.settings.jpeg_quality)
        self.prepare_seconds += time.perf_counter() - started
        self.prepare_attempts += 1
        if frame is None or len(frame) > self._server.settings.max_frame_bytes:
            return True
        started = time.perf_counter()
        if not self._transport.send_encoded_jpeg(frame):
            self.close(send_end=False)
            return False
        self.send_seconds += time.perf_counter() - started
        return True

    def send_bgr(self, image) -> bool:
        if self._closed:
            return False
        started = time.perf_counter()
        frame = prepare_game_bgr(image, quality=self._server.settings.jpeg_quality)
        self.prepare_seconds += time.perf_counter() - started
        self.prepare_attempts += 1
        if frame is None or len(frame) > self._server.settings.max_frame_bytes:
            return True
        started = time.perf_counter()
        if not self._transport.send_encoded_jpeg(frame):
            self.close(send_end=False)
            return False
        self.send_seconds += time.perf_counter() - started
        return True

    def close(self, *, send_end: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        sent_frames = self._transport.sent_frames
        self._transport.close(send_end=send_end)
        self._server._log(
            "info", f"游戏 TFT 持续流结束: sent_frames={sent_frames}"
        )


class GameTftStreamServer(TftPreviewServer):
    """Opt-in subclass adding a persistent stream without changing legacy API."""

    def open_jpeg_stream(self, *, fps: int) -> GameTftStream | None:
        transport = self.open_persistent_stream(fps=fps)
        if transport is None:
            return None
        return GameTftStream(self, transport)


__all__ = [
    "GameTftStream", "GameTftStreamServer", "prepare_game_bgr", "prepare_game_jpeg",
]
