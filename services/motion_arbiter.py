"""Pure motor-command validation and priority arbitration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


MOTOR_OUTPUT_TOPIC = "/motor_cmd"
MOTOR_AUTONOMY_TOPIC = "/motor_cmd/autonomy"
MOTOR_TRACKING_TOPIC = "/motor_cmd/tracking"
MOTOR_JOYSTICK_TOPIC = "/motor_cmd/joystick"

SOURCE_AUTONOMY = "autonomy"
SOURCE_TRACKING = "tracking"
SOURCE_JOYSTICK = "joystick"
SOURCE_PRIORITY = (SOURCE_JOYSTICK, SOURCE_TRACKING, SOURCE_AUTONOMY)
SOURCE_TOPICS = {
    SOURCE_AUTONOMY: MOTOR_AUTONOMY_TOPIC,
    SOURCE_TRACKING: MOTOR_TRACKING_TOPIC,
    SOURCE_JOYSTICK: MOTOR_JOYSTICK_TOPIC,
}

COMMAND_TIMEOUT_SEC = 0.3
STOP_COMMAND = {
    "left": {"action": 0, "throttle": 0},
    "right": {"action": 0, "throttle": 0},
}


def normalize_motor_command(payload: object) -> dict[str, dict[str, int]] | None:
    """Validate and normalize the two-track motor command protocol."""
    if not isinstance(payload, dict):
        return None
    normalized = {}
    for side in ("left", "right"):
        motor = payload.get(side)
        if not isinstance(motor, dict):
            return None
        action = motor.get("action")
        throttle = motor.get("throttle")
        if isinstance(action, bool) or not isinstance(action, int) or action not in (0, 1, 2):
            return None
        if isinstance(throttle, bool) or not isinstance(throttle, (int, float)):
            return None
        if not 0 <= float(throttle) <= 100:
            return None
        if action == 0:
            throttle = 0
        normalized[side] = {"action": action, "throttle": int(throttle)}
    return normalized


@dataclass
class _CommandState:
    command: dict[str, dict[str, int]]
    received_at: float


class MotionArbiter:
    """Choose the freshest command from the highest-priority active source."""

    def __init__(
        self,
        *,
        timeout_sec: float = COMMAND_TIMEOUT_SEC,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.timeout_sec = float(timeout_sec)
        self._clock = clock
        self._commands: dict[str, _CommandState] = {}

    def update(self, source: str, payload: object) -> bool:
        if source not in SOURCE_TOPICS:
            return False
        command = normalize_motor_command(payload)
        if command is None:
            return False
        self._commands[source] = _CommandState(command, self._clock())
        return True

    def select(self) -> tuple[str, dict[str, dict[str, int]]]:
        source, command, _deadline = self.select_with_deadline()
        return source, command

    def select_with_deadline(
        self,
    ) -> tuple[str, dict[str, dict[str, int]], float | None]:
        """Return the selected command and its monotonic expiry deadline."""
        now = self._clock()
        for source in SOURCE_PRIORITY:
            state = self._commands.get(source)
            if state is not None and now - state.received_at <= self.timeout_sec:
                return source, state.command, state.received_at + self.timeout_sec
        return "failsafe", STOP_COMMAND, None
