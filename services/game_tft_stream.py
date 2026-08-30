"""Persistent chest-TFT stream used only by the isolated game probe."""

from __future__ import annotations

from services.tft_preview_server import (
    JPEG_FRAME,
    STREAM_END,
    STREAM_START_MESSAGE,
    TftPreviewServer,
    encode_stream_start,
)


def prepare_game_bgr(image, *, quality: int = 70) -> bytes | None:
    """Fit an upright BGR game frame to the TFT without cropping or rotation."""
    try:
        import cv2
    except ImportError:
        return None
    if image is None or getattr(image, "size", 0) == 0:
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


def prepare_game_jpeg(jpeg: bytes, *, quality: int = 70) -> bytes | None:
    """JPEG compatibility path for virtual-display sources."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    raw = bytes(jpeg or b"")
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        return None
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return prepare_game_bgr(image, quality=quality)


class GameTftStream:
    PERSISTENT_DURATION = 0xFFFFFFFF

    def __init__(
        self,
        server: "GameTftStreamServer",
        client,
        sequence: int,
    ) -> None:
        self._server = server
        self._client = client
        self._stream_sequence = sequence
        self._frame_index = 0
        self._closed = False

    def send_jpeg(self, jpeg: bytes) -> bool:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return True
        raw = bytes(jpeg or b"")
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            return True
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return self.send_bgr(image)

    def send_bgr(self, image) -> bool:
        if self._closed:
            return False
        frame = prepare_game_bgr(image, quality=self._server.settings.jpeg_quality)
        if frame is None or len(frame) > self._server.settings.max_frame_bytes:
            return True
        sequence = ((self._stream_sequence & 0xFFFF) << 16) | (self._frame_index & 0xFFFF)
        try:
            self._server._send_packet(self._client, JPEG_FRAME, sequence, frame)
        except (ConnectionError, OSError):
            self.close(send_end=False)
            return False
        self._frame_index += 1
        return True

    def close(self, *, send_end: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if send_end:
            try:
                self._server._send_packet(
                    self._client, STREAM_END, self._stream_sequence, b""
                )
            except (ConnectionError, OSError):
                pass
        self._server._log(
            "info", f"游戏 TFT 持续流结束: sent_frames={self._frame_index}"
        )
        self._server._stream_lock.release()


class GameTftStreamServer(TftPreviewServer):
    """Opt-in subclass adding a persistent stream without changing legacy API."""

    def open_jpeg_stream(self, *, fps: int) -> GameTftStream | None:
        if not self._stream_lock.acquire(blocking=False):
            return None
        client = self._verified_client()
        if client is None:
            self._stream_lock.release()
            return None
        stream_sequence = self._next_stream_sequence()
        target_fps = min(30, max(1, int(fps)))
        try:
            self._send_packet(
                client,
                STREAM_START_MESSAGE,
                stream_sequence,
                encode_stream_start(GameTftStream.PERSISTENT_DURATION, 0, target_fps),
            )
        except (ConnectionError, OSError):
            self._stream_lock.release()
            return None
        return GameTftStream(self, client, stream_sequence)


__all__ = ["GameTftStream", "GameTftStreamServer", "prepare_game_bgr", "prepare_game_jpeg"]
