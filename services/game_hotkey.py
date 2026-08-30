"""Game-mode hotkey recognition independent from evdev and ROS."""

from __future__ import annotations

import time
from typing import Callable


class StartSelectHold:
    """Emit one event when Start and Select remain down for the hold interval."""

    def __init__(self, *, hold_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.hold_seconds = float(hold_seconds)
        self._clock = clock
        self._start_down = False
        self._select_down = False
        self._both_since: float | None = None
        self._fired = False

    def set_start(self, down: bool) -> None:
        self._start_down = bool(down)
        self._refresh()

    def set_select(self, down: bool) -> None:
        self._select_down = bool(down)
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
        if self._start_down and self._select_down:
            if self._both_since is None:
                self._both_since = self._clock()
                self._fired = False
            return
        self._both_since = None
        self._fired = False


__all__ = ["StartSelectHold"]
