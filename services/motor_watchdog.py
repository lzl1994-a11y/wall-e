"""Reusable deadman timer for the final motor hardware backends."""

from __future__ import annotations

import time
from typing import Callable


MOTOR_WATCHDOG_TIMEOUT_SEC = 0.3
MOTOR_WATCHDOG_CHECK_INTERVAL_SEC = 0.05


class MotorWatchdog:
    def __init__(
        self,
        *,
        timeout_sec: float = MOTOR_WATCHDOG_TIMEOUT_SEC,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.timeout_sec = float(timeout_sec)
        self._clock = clock
        self._last_refresh = clock()
        self._tripped = False

    def refresh(self) -> bool:
        """Refresh the heartbeat and return whether this recovers a tripped timer."""
        recovered = self._tripped
        self._last_refresh = self._clock()
        self._tripped = False
        return recovered

    def poll(self) -> bool:
        """Return True once when the current heartbeat window expires."""
        if self._tripped:
            return False
        if self._clock() - self._last_refresh < self.timeout_sec:
            return False
        self._tripped = True
        return True
