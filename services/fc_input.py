"""FC controller mapping and an X11 input relay for isolated game sessions."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import select
import struct
import threading
from dataclasses import dataclass
from typing import Any, Callable


EV_KEY = 1
EV_ABS = 3
ABS_HAT0X = 16
ABS_HAT0Y = 17
BTN_SOUTH = 304
BTN_EAST = 305
BTN_NORTH = 307
BTN_WEST = 308
BTN_SELECT = 314
BTN_START = 315


@dataclass(frozen=True)
class FcKeyChange:
    key: str
    down: bool


_BUTTON_KEYS = {
    BTN_SOUTH: "F",       # physical A -> FC A
    BTN_NORTH: "F",       # physical Y -> FC A
    BTN_EAST: "D",        # physical B -> FC B
    BTN_WEST: "D",        # physical X -> FC B
    BTN_SELECT: "S",
    BTN_START: "Return",
}

_HAT_KEYS = {
    ABS_HAT0X: ("KP_4", "KP_6"),
    ABS_HAT0Y: ("KP_8", "KP_2"),
}

_JS_BUTTON_KEYS = {
    0: "F", 3: "F",  # physical A/Y -> FC A
    1: "D", 2: "D",  # physical B/X -> FC B
    8: "S", 9: "Return",
}
_JS_HAT_AXES = {6: ("KP_4", "KP_6"), 7: ("KP_8", "KP_2")}
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80
_JS_EVENT = struct.Struct("<IhBB")


class FcInputMapper:
    """Translate the confirmed red-mode evdev layout to FCEUX default keys."""

    def __init__(self) -> None:
        self._hat_values = {code: 0 for code in _HAT_KEYS}

    def translate(self, event_type: int, code: int, value: int) -> list[FcKeyChange]:
        if event_type == EV_KEY and code in _BUTTON_KEYS:
            if value not in (0, 1):
                return []
            return [FcKeyChange(_BUTTON_KEYS[code], value == 1)]
        if event_type != EV_ABS or code not in _HAT_KEYS:
            return []

        value = -1 if value < 0 else 1 if value > 0 else 0
        previous = self._hat_values[code]
        if value == previous:
            return []
        negative_key, positive_key = _HAT_KEYS[code]
        changes: list[FcKeyChange] = []
        if previous < 0:
            changes.append(FcKeyChange(negative_key, False))
        elif previous > 0:
            changes.append(FcKeyChange(positive_key, False))
        if value < 0:
            changes.append(FcKeyChange(negative_key, True))
        elif value > 0:
            changes.append(FcKeyChange(positive_key, True))
        self._hat_values[code] = value
        return changes


class FcJoystickMapper:
    """Map the confirmed joystick API layout without evdev sync/analog noise."""

    def __init__(self) -> None:
        self._hat_values = {axis: 0 for axis in _JS_HAT_AXES}

    def translate(self, event_type: int, number: int, value: int) -> list[FcKeyChange]:
        event_type &= ~_JS_EVENT_INIT
        if event_type == _JS_EVENT_BUTTON and number in _JS_BUTTON_KEYS:
            return [FcKeyChange(_JS_BUTTON_KEYS[number], value != 0)]
        if event_type != _JS_EVENT_AXIS or number not in _JS_HAT_AXES:
            return []
        value = -1 if value < 0 else 1 if value > 0 else 0
        previous = self._hat_values[number]
        if value == previous:
            return []
        negative_key, positive_key = _JS_HAT_AXES[number]
        changes: list[FcKeyChange] = []
        if previous < 0:
            changes.append(FcKeyChange(negative_key, False))
        elif previous > 0:
            changes.append(FcKeyChange(positive_key, False))
        if value < 0:
            changes.append(FcKeyChange(negative_key, True))
        elif value > 0:
            changes.append(FcKeyChange(positive_key, True))
        self._hat_values[number] = value
        return changes


class XTestKeySink:
    """Inject key transitions into one X11 display through the XTEST extension."""

    def __init__(self, display: str) -> None:
        x11_name = ctypes.util.find_library("X11") or "libX11.so.6"
        xtst_name = ctypes.util.find_library("Xtst") or "libXtst.so.6"
        self._x11 = ctypes.CDLL(x11_name)
        self._xtst = ctypes.CDLL(xtst_name)
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self._x11.XStringToKeysym.restype = ctypes.c_ulong
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XFlush.argtypes = [ctypes.c_void_p]
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]
        self._xtst.XTestFakeKeyEvent.restype = ctypes.c_int
        self._display = self._x11.XOpenDisplay(display.encode("ascii"))
        if not self._display:
            raise RuntimeError(f"cannot open X11 display {display}")
        self._pressed: set[str] = set()

    def set_key(self, key: str, down: bool) -> None:
        keysym = self._x11.XStringToKeysym(key.encode("ascii"))
        keycode = self._x11.XKeysymToKeycode(self._display, keysym)
        if not keycode:
            raise RuntimeError(f"X11 has no keycode for {key}")
        if not self._xtst.XTestFakeKeyEvent(self._display, keycode, int(down), 0):
            raise RuntimeError(f"failed to inject X11 key {key}")
        self._x11.XFlush(self._display)
        if down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)

    def close(self) -> None:
        display = self._display
        if not display:
            return
        for key in tuple(self._pressed):
            self.set_key(key, False)
        self._x11.XCloseDisplay(display)
        self._display = None


class FcControllerRelay:
    """Exclusively relay one physical controller to an X11 FCEUX session."""

    def __init__(
        self,
        device_path: str,
        sink: XTestKeySink,
        *,
        mapper: FcInputMapper | None = None,
        device_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._device_path = device_path
        self._sink = sink
        self._mapper = mapper or FcInputMapper()
        self._device_factory = device_factory
        self._device: Any = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.error: Exception | None = None

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("FC controller relay is already running")
        factory = self._device_factory
        if factory is None:
            from evdev import InputDevice

            factory = InputDevice
        device = factory(self._device_path)
        device.grab()
        self._device = device
        self._stop.clear()
        self.error = None
        self._worker = threading.Thread(target=self._read_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        device = self._device
        self._device = None
        if device is not None:
            try:
                device.ungrab()
            except OSError:
                pass
            device.close()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self._worker = None
        self._sink.close()

    def _read_loop(self) -> None:
        device = self._device
        if device is None:
            return
        try:
            for event in device.read_loop():
                if self._stop.is_set():
                    return
                for change in self._mapper.translate(event.type, event.code, event.value):
                    self._sink.set_key(change.key, change.down)
        except OSError as exc:
            if not self._stop.is_set():
                self.error = exc


class FcJoystickRelay:
    """Read the lean joystick stream while evdev remains exclusively grabbed."""

    def __init__(
        self,
        joystick_path: str,
        sink,
        *,
        grab_device_path: str = "/dev/input/event2",
        mapper: FcJoystickMapper | None = None,
    ) -> None:
        self._joystick_path = joystick_path
        self._grab_device_path = grab_device_path
        self._sink = sink
        self._mapper = mapper or FcJoystickMapper()
        self._grab_device: Any = None
        self._fd: int | None = None
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("FC joystick relay is already running")
        from evdev import InputDevice

        grab_device = InputDevice(self._grab_device_path)
        try:
            grab_device.grab()
            fd = os.open(self._joystick_path, os.O_RDONLY | os.O_NONBLOCK)
        except Exception:
            try:
                grab_device.ungrab()
            finally:
                grab_device.close()
            raise
        self._grab_device = grab_device
        self._fd = fd
        self._stop.clear()
        self.error = None
        self._worker = threading.Thread(target=self._read_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self._worker = None
        grab_device, self._grab_device = self._grab_device, None
        if grab_device is not None:
            try:
                grab_device.ungrab()
            except OSError:
                pass
            grab_device.close()
        self._sink.close()

    def _read_loop(self) -> None:
        fd = self._fd
        if fd is None:
            return
        remainder = b""
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                data = remainder + os.read(fd, 4096)
                size = len(data) - (len(data) % _JS_EVENT.size)
                remainder = data[size:]
                for offset in range(0, size, _JS_EVENT.size):
                    _, value, event_type, number = _JS_EVENT.unpack_from(data, offset)
                    for change in self._mapper.translate(event_type, number, value):
                        self._sink.set_key(change.key, change.down)
        except OSError as exc:
            if not self._stop.is_set():
                self.error = exc


__all__ = [
    "FcControllerRelay", "FcInputMapper", "FcJoystickMapper", "FcJoystickRelay",
    "FcKeyChange", "XTestKeySink",
]
