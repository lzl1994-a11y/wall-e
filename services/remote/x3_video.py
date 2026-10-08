"""Bounded X3 JPEG -> NV12 -> Baseline H264 using the public media SDK."""
from __future__ import annotations

import ctypes
from io import BytesIO
from pathlib import Path
import threading


_owner = threading.Lock()
MAX_JPEG_BYTES = 2 * 1024 * 1024


class HardwareVideoError(RuntimeError):
    pass


def require_codec_library(root: Path) -> Path:
    library = root / "build/wali_webrtc_codec/libwali_webrtc_codec.so"
    sources = [root / "cpp_nodes/wali_webrtc_codec/src/codec.c",
               root / "cpp_nodes/wali_webrtc_codec/CMakeLists.txt"]
    if not library.is_file() or any(s.stat().st_mtime_ns > library.stat().st_mtime_ns for s in sources):
        raise HardwareVideoError("X3 video codec missing/stale; run tools/build_webrtc_codec.sh before launch")
    return library


class X3VideoCodec:
    """One owner per process; caller serializes encode and close on one worker."""

    def __init__(self, library: Path):
        if not _owner.acquire(blocking=False):
            raise HardwareVideoError("X3 video codec already in use")
        self._closed = False
        self._lib = None
        self._profile = None
        self._source_size = None
        self._decoded = None
        self._output = ctypes.create_string_buffer(MAX_JPEG_BYTES)
        try:
            self._lib = ctypes.CDLL(str(library))
            self._lib.walle_init.argtypes = []
            self._lib.walle_close.argtypes = []
            self._lib.walle_encoder_close.argtypes = []
            self._lib.walle_encoder_open.argtypes = [ctypes.c_int] * 4
            self._lib.walle_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self._lib.walle_encode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_int64]
            for name in ["walle_init", "walle_close", "walle_encoder_close", "walle_encoder_open",
                         "walle_decode", "walle_encode"]:
                getattr(self._lib, name).restype = ctypes.c_int
            self._check(self._lib.walle_init(), "initialize")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise HardwareVideoError(f"X3 video {operation} failed: {result}")

    def encode(self, jpeg: bytes, width: int, height: int, fps: int, pts: int) -> bytes:
        if self._closed:
            raise HardwareVideoError("X3 video codec is closed")
        if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("invalid JPEG size")
        # Read JPEG header only; the JPU does the pixel decoding.
        from PIL import Image
        with Image.open(BytesIO(jpeg)) as image:
            if image.format != "JPEG":
                raise ValueError("X3 video input must be JPEG")
            source_width, source_height = image.size
        if (source_width <= 0 or source_height <= 0 or source_width > 1920 or source_height > 1080
                or source_width % 2 or source_height % 2):
            raise ValueError("unsupported JPEG dimensions")
        if self._source_size is not None and self._source_size != (source_width, source_height):
            raise HardwareVideoError("camera dimensions changed during hardware session")
        self._source_size = (source_width, source_height)
        if self._decoded is None:
            self._decoded = ctypes.create_string_buffer(source_width * source_height * 3 // 2)
        self._check(self._lib.walle_decode(jpeg, len(jpeg), self._decoded, len(self._decoded),
                                           source_width, source_height), "JPEG decode")
        pixels = self._decoded.raw
        if (width, height) != self._source_size:
            import av
            import numpy as np
            source = np.frombuffer(pixels, dtype=np.uint8).reshape((source_height * 3 // 2, source_width))
            frame = av.VideoFrame.from_ndarray(source, format="nv12")
            # Auto scaling creates a CPU-sized pool on every fresh VideoFrame.
            # One scaling thread is faster for these small frames on X3 and
            # leaves the realtime media loop room to receive audio.
            pixels = frame.reformat(width=width, height=height, format="nv12", threads=1).to_ndarray().tobytes()
        profile = (width, height, fps)
        if profile != self._profile:
            self._check(self._lib.walle_encoder_close(), "encoder close for profile change")
            self._check(self._lib.walle_encoder_open(width, height, fps, 150 if fps == 5 else 500),
                        "encoder open")
            self._profile = profile
        size = self._lib.walle_encode(pixels, len(pixels), self._output, len(self._output), pts // 90)
        if size <= 0 or size > len(self._output):
            raise HardwareVideoError(f"X3 video H264 encode failed: {size}")
        return self._output.raw[:size]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._lib is not None:
                self._check(self._lib.walle_close(), "close")
        finally:
            _owner.release()
