"""Pure helpers for differential-drive commands and motor direction polarity."""

from __future__ import annotations


def speed_to_motor(speed: float) -> dict[str, int]:
    """Convert a normalized signed speed into the motor command protocol."""
    speed = max(-1.0, min(1.0, float(speed)))
    action = 1 if speed > 0 else (2 if speed < 0 else 0)
    return {"action": action, "throttle": int(abs(speed) * 100)}


def mix_differential_drive(forward: float, turn: float) -> dict[str, dict[str, int]]:
    """Arcade mix: positive forward, negative left turn, positive right turn."""
    left_speed = max(-1.0, min(1.0, float(forward) + float(turn)))
    right_speed = max(-1.0, min(1.0, float(forward) - float(turn)))
    return {
        "left": speed_to_motor(left_speed),
        "right": speed_to_motor(right_speed),
    }


def apply_direction_inversion(action: int, inverted: bool) -> int:
    """Swap forward/reverse for a physically mirrored motor."""
    if not inverted:
        return action
    if action == 1:
        return 2
    if action == 2:
        return 1
    return 0


def motor_inversion_flags(motors: object) -> dict[str, bool]:
    """Read logical left/right direction polarity from config.motors."""
    flags = {"left": False, "right": False}
    if not isinstance(motors, list):
        return flags
    for motor in motors:
        if not isinstance(motor, dict):
            continue
        side = {"track_l": "left", "track_r": "right"}.get(motor.get("name"))
        if side:
            flags[side] = bool(motor.get("invert_direction", False))
    return flags
