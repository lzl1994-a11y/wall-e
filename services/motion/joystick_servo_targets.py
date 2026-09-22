"""Compute manual servo targets for one joystick control tick."""

from __future__ import annotations

from typing import Any, Mapping

# Default joystick axis indices matching standard controller mapping
AXIS_RX: int = 3
AXIS_RY: int = 4
AXIS_L2: int = 2
AXIS_R2: int = 5


def compute_joystick_servo_targets(
    axes: Mapping[int, float],
    auto_timers: Mapping[str, float],
    now: float,
    neck_kinematics: Any,
) -> dict[str, int]:
    """Compute servo target dictionary from normalized axes and countdown timers.

    Calculates targets for head_yaw, neck pitch coupling, eyes, arms, and eyebrows
    strictly preserving existing order and mathematical formulas.
    """
    rx = axes[AXIS_RX]
    ry = axes[AXIS_RY]
    l2 = axes[AXIS_L2]
    r2 = axes[AXIS_R2]

    targets: dict[str, int] = {}

    # 1. 头部方向 (右摇杆 X)
    targets["head_yaw"] = int(5000 - rx * 2600)

    # 2. 脖子俯仰：由已加载的 NeckKinematics 联动解算 (右摇杆 Y)
    targets.update(neck_kinematics.targets(ry))

    # 3. 眼睛扳机 (L2 / R2)
    targets["eye_l"] = int(7500 - l2 * 2500)
    targets["eye_r"] = int(2000 + r2 * 2000)

    # 4. 手臂与眉毛 (基于截止时间的倒计时逻辑)
    targets["arm_l"] = 6000 if now < auto_timers["arm_l"] else 2000
    targets["arm_r"] = 4000 if now < auto_timers["arm_r"] else 8000
    targets["eyebrow_l"] = 5700 if now < auto_timers["eyebrow_l"] else 8000
    targets["eyebrow_r"] = 4200 if now < auto_timers["eyebrow_r"] else 1920

    return targets
