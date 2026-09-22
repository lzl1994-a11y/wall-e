"""Game-mode hotkey recognition independent from evdev and ROS."""

from __future__ import annotations

import time
from typing import Callable


class ButtonChordHold:
    """Emit one event when two buttons remain down for the hold interval."""

    def __init__(self, *, hold_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.hold_seconds = float(hold_seconds)
        self._clock = clock
        self._first_down = False
        self._second_down = False
        self._both_since: float | None = None
        self._fired = False

    def set_first(self, down: bool) -> None:
        self._first_down = bool(down)
        self._refresh()

    def set_second(self, down: bool) -> None:
        self._second_down = bool(down)
        self._refresh()

    def poll(self) -> bool:
        """Return true once for each completed long press."""
        self._refresh()
        if self._fired or self._both_since is None:
            return False
        if self._clock() - self._both_since < self.hold_seconds:
            return False
        self._fired = True
        return True

    def _refresh(self) -> None:
        if self._first_down and self._second_down:
            if self._both_since is None:
                self._both_since = self._clock()
                self._fired = False
            return
        self._both_since = None
        self._fired = False


class ButtonHold:
    """Emit one event when one button remains down for the hold interval."""

    def __init__(self, *, hold_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.hold_seconds = float(hold_seconds)
        self._clock = clock
        self._down_since: float | None = None
        self._fired = False

    def set_down(self, down: bool) -> None:
        if down:
            if self._down_since is None:
                self._down_since = self._clock()
                self._fired = False
            return
        self._down_since = None
        self._fired = False

    def poll(self) -> bool:
        """Return true once for each completed long press."""
        if self._fired or self._down_since is None:
            return False
        if self._clock() - self._down_since < self.hold_seconds:
            return False
        self._fired = True
        return True


class StartSelectHold(ButtonChordHold):
    """Backward-compatible named wrapper for the original hotkey."""

    def set_start(self, down: bool) -> None:
        self.set_first(down)

    def set_select(self, down: bool) -> None:
        self.set_second(down)


__all__ = ["ButtonChordHold", "ButtonHold", "StartSelectHold"]
