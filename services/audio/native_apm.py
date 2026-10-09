"""ctypes releases the GIL while the existing system APM processes audio."""
from __future__ import annotations

import ctypes
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[2] / "build/wali_audio_apm/libwali_audio_apm.so"


class NativeEchoApm:
    def __init__(self, pre_gain_db):
        lib = ctypes.CDLL(str(LIBRARY))
        lib.wali_apm_create.argtypes = [ctypes.c_float]
        lib.wali_apm_create.restype = ctypes.c_void_p
        lib.wali_apm_destroy.argtypes = [ctypes.c_void_p]
        lib.wali_apm_destroy.restype = None
        lib.wali_apm_render.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.wali_apm_capture.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int]
        self._lib = lib
        self._context = lib.wali_apm_create(pre_gain_db)
        if not self._context:
            raise RuntimeError("native WebRTC AEC initialization failed")

    @staticmethod
    def _samples(pcm):
        if not pcm or len(pcm) % 960:
            raise ValueError("AEC requires whole 10 ms frames at 48 kHz mono")
        return len(pcm) // 2

    def render(self, pcm):
        rc = self._lib.wali_apm_render(self._context, pcm, self._samples(pcm))
        if rc:
            raise RuntimeError(f"native AEC render failed: {rc}")

    def capture(self, pcm, delay_ms):
        output = ctypes.create_string_buffer(len(pcm))
        rc = self._lib.wali_apm_capture(self._context, pcm, output, self._samples(pcm), delay_ms)
        if rc:
            raise RuntimeError(f"native AEC capture failed: {rc}")
        return output.raw

    def close(self):
        if self._context:
            self._lib.wali_apm_destroy(self._context)
            self._context = None
