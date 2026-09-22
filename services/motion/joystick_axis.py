"""Pure normalization helpers for joystick and trigger EV_ABS events."""

from __future__ import annotations


def normalize_axis_value(
    raw_value: float,
    min_value: float,
    max_value: float,
    *,
    deadzone: float = 0.0,
    is_trigger: bool = False,
    invert_y: bool = False,
) -> float:
    """Normalize a raw axis value using device calibration without upper clipping.

    A zero-width or inverted device range returns a neutral value so a bad
    calibration cannot crash the joystick reader or produce a large command.

    For triggers:
        n_val = max(0, raw_value - min_value) / max(1, max_value - min_value)
        Does not apply deadzone, does not invert, does not clip upper bound.

    For sticks:
        mid = (min_value + max_value) / 2.0
        n_val = (raw_value - mid) / float(max_value - mid)
        If abs(n_val) < deadzone: n_val = 0.0
        If invert_y: n_val = -n_val
    """
    if max_value <= min_value:
        return 0.0

    if is_trigger:
        return max(0, raw_value - min_value) / max(1, max_value - min_value)

    mid = (min_value + max_value) / 2.0
    n_val = (raw_value - mid) / float(max_value - mid)

    if abs(n_val) < deadzone:
        n_val = 0.0

    if invert_y:
        n_val = -n_val

    return n_val
