"""Minimal libretro frontend for an FC core with direct video-frame callbacks."""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Callable


RETRO_ENVIRONMENT_SET_PIXEL_FORMAT = 10
RETRO_PIXEL_FORMAT_XRGB8888 = 1
RETRO_DEVICE_JOYPAD = 1
RETRO_DEVICE_ID_JOYPAD_B = 0
RETRO_DEVICE_ID_JOYPAD_A = 8
RETRO_DEVICE_ID_JOYPAD_SELECT = 2
RETRO_DEVICE_ID_JOYPAD_START = 3
RETRO_DEVICE_ID_JOYPAD_UP = 4
RETRO_DEVICE_ID_JOYPAD_DOWN = 5
RETRO_DEVICE_ID_JOYPAD_LEFT = 6
RETRO_DEVICE_ID_JOYPAD_RIGHT = 7


class _GameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]


EnvironmentCallback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VideoCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t
)
AudioSampleCallback = ctypes.CFUNCTYPE(None, ctypes.c_short, ctypes.c_short)
AudioBatchCallback = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_short), ctypes.c_size_t
)
InputPollCallback = ctypes.CFUNCTYPE(None)
InputStateCallback = ctypes.CFUNCTYPE(
    ctypes.c_short, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
)


class LibretroJoypad:
    """Accept the FC adapter's key names and expose libretro joypad state."""

    _KEYS = {
        "F": RETRO_DEVICE_ID_JOYPAD_A,
        "D": RETRO_DEVICE_ID_JOYPAD_B,
        "S": RETRO_DEVICE_ID_JOYPAD_SELECT,
        "Return": RETRO_DEVICE_ID_JOYPAD_START,
        "KP_8": RETRO_DEVICE_ID_JOYPAD_UP,
        "KP_2": RETRO_DEVICE_ID_JOYPAD_DOWN,
        "KP_4": RETRO_DEVICE_ID_JOYPAD_LEFT,
        "KP_6": RETRO_DEVICE_ID_JOYPAD_RIGHT,
    }

    def __init__(self) -> None:
        self._pressed: set[int] = set()

    def set_key(self, key: str, down: bool) -> None:
        control = self._KEYS.get(key)
        if control is None:
            return
        if down:
            self._pressed.add(control)
        else:
            self._pressed.discard(control)

    def state(self, port: int, device: int, index: int, control: int) -> int:
        if port != 0 or device != RETRO_DEVICE_JOYPAD or index != 0:
            return 0
        return int(control in self._pressed)

    def close(self) -> None:
        """Match the relay sink lifecycle; no external resource is owned."""
        self._pressed.clear()


class LibretroFc:
    """Load a libretro FC core and pass unbuffered XRGB frames to a callback."""

    def __init__(
        self,
        core_path: str | Path,
        *,
        on_frame: Callable[[bytes, int, int, int], None],
        audio_sink=None,
    ):
        self._core = ctypes.CDLL(str(core_path))
        self._on_frame = on_frame
        self._audio_sink = audio_sink
        self.joypad = LibretroJoypad()
        self._configure_api()
        self._callbacks = (
            EnvironmentCallback(self._environment),
            VideoCallback(self._video),
            AudioSampleCallback(self._audio_sample),
            AudioBatchCallback(self._audio_batch),
            InputPollCallback(lambda: None),
            InputStateCallback(self.joypad.state),
        )
        self._initialized = False

    def _configure_api(self) -> None:
        for name, callback_type in (
            ("retro_set_environment", EnvironmentCallback),
            ("retro_set_video_refresh", VideoCallback),
            ("retro_set_audio_sample", AudioSampleCallback),
            ("retro_set_audio_sample_batch", AudioBatchCallback),
            ("retro_set_input_poll", InputPollCallback),
            ("retro_set_input_state", InputStateCallback),
        ):
            function = getattr(self._core, name)
            function.argtypes = [callback_type]
        self._core.retro_load_game.argtypes = [ctypes.POINTER(_GameInfo)]
        self._core.retro_load_game.restype = ctypes.c_bool

    def load(self, rom_path: str | Path) -> None:
        callbacks = self._callbacks
        self._core.retro_set_environment(callbacks[0])
        self._core.retro_set_video_refresh(callbacks[1])
        self._core.retro_set_audio_sample(callbacks[2])
        self._core.retro_set_audio_sample_batch(callbacks[3])
        self._core.retro_set_input_poll(callbacks[4])
        self._core.retro_set_input_state(callbacks[5])
        self._core.retro_init()
        self._initialized = True
        info = _GameInfo(str(rom_path).encode(), None, 0, None)
        if not self._core.retro_load_game(ctypes.byref(info)):
            raise RuntimeError(f"libretro core could not load {rom_path}")

    def run_for(self, seconds: float, *, fps: float = 60.0) -> None:
        deadline = time.monotonic() + max(0.1, seconds)
        period = 1.0 / max(1.0, fps)
        while time.monotonic() < deadline:
            started = time.monotonic()
            self._core.retro_run()
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def run_until(self, should_stop: Callable[[], bool], *, fps: float = 60.0) -> None:
        """Run in real time until the session owner requests teardown."""
        period = 1.0 / max(1.0, fps)
        while not should_stop():
            started = time.monotonic()
            self._core.retro_run()
            delay = period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def close(self) -> None:
        if self._initialized:
            try:
                self._core.retro_unload_game()
            finally:
                self._core.retro_deinit()
                self._initialized = False

    def _environment(self, command: int, data: int) -> bool:
        if command != RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            return False
        pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_int))
        return bool(pixel_format and pixel_format.contents.value == RETRO_PIXEL_FORMAT_XRGB8888)

    def _video(self, data: int, width: int, height: int, pitch: int) -> None:
        if not data or not width or not height:
            return
        self._on_frame(ctypes.string_at(data, pitch * height), width, height, pitch)

    def _audio_sample(self, left: int, right: int) -> None:
        if self._audio_sink is not None:
            self._audio_sink.push_sample(left, right)

    def _audio_batch(self, data, frames: int) -> int:
        if self._audio_sink is not None:
            self._audio_sink.push_batch(data, frames)
        return frames


__all__ = ["LibretroFc", "LibretroJoypad"]
