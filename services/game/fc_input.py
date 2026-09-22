"""FC controller mapping and an X11 input relay for isolated game sessions."""

from __future__ import annotations

import ctypes
import ctypes.util
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
        self._sink_lock = threading.Lock()
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
        with self._sink_lock:
            self._sink.close()

    def switch_sink(self, sink) -> None:
        """Route subsequent controller changes to another lifecycle-neutral sink."""
        with self._sink_lock:
            previous = self._sink
            self._sink = sink
            if previous is not sink:
                previous.close()

    def _read_loop(self) -> None:
        device = self._device
        if device is None:
            return
        try:
            for event in device.read_loop():
                if self._stop.is_set():
                    return
                for change in self._mapper.translate(event.type, event.code, event.value):
                    with self._sink_lock:
                        self._sink.set_key(change.key, change.down)
        except OSError as exc:
            if not self._stop.is_set():
                self.error = exc


__all__ = ["FcControllerRelay", "FcInputMapper", "FcKeyChange", "XTestKeySink"]
