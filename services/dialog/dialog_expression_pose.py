"""Resolve servo targets and step sizes for dialogue expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from services.motion.servo_motion_config import resolve_servo_target

DEFAULT_STEP_SIZE: float = 24.0
DEFAULT_INTENSITY_FACTOR: float = 0.6
EXPRESSION_INTENSITY_FACTORS: dict[str, float] = {
    "low": 0.6,
    "medium": 0.85,
    "high": 1.0,
}


@dataclass(frozen=True)
class DialogExpressionPose:
    """Resolved servo targets and execution step size for an expression pose."""

    targets: dict[str, int]
    step_size: float


def resolve_pose_targets(
    pose: Mapping[str, Any] | None,
    servos: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Resolve raw pose target definitions to bounded integer PWM values."""
    if not isinstance(pose, Mapping) or not isinstance(servos, Mapping):
        return {}

    raw_targets = pose.get("targets")
    if not isinstance(raw_targets, Mapping):
        return {}

    resolved: dict[str, int] = {}
    for name, raw_target in raw_targets.items():
        servo = servos.get(name)
        if servo is None:
            continue
        target = resolve_servo_target(servo, raw_target)
        if target is not None:
            resolved[name] = target
    return resolved


def interpolate_expression_targets(
    targets: Mapping[str, int],
    neutral_targets: Mapping[str, int] | None,
    intensity: str | None = None,
) -> dict[str, int]:
    """Interpolate non-neutral expression targets against neutral targets."""
    if not neutral_targets:
        return dict(targets)

    factor = EXPRESSION_INTENSITY_FACTORS.get(intensity, DEFAULT_INTENSITY_FACTOR)
    scaled: dict[str, int] = {}
    for name, target in targets.items():
        neutral_target = neutral_targets.get(name, target)
        scaled[name] = int(round(
            neutral_target + (target - neutral_target) * factor
        ))
    return scaled


def resolve_dialog_expression_pose(
    expression_poses: Mapping[str, Any] | None,
    servos: Mapping[str, Any] | None,
    expression: str | None,
    intensity: str | None = "low",
    default_step: float | None = DEFAULT_STEP_SIZE,
    neutral_targets: Mapping[str, int] | None = None,
) -> DialogExpressionPose:
    """Compute servo target dictionary and step size for a dialogue expression.

    If ``expression`` is missing or its configuration is empty, falls back to
    the ``neutral`` pose configuration. Non-neutral expressions are scaled by
    intensity relative to neutral targets.
    """
    poses = expression_poses if isinstance(expression_poses, Mapping) else {}
    servos_map = servos if isinstance(servos, Mapping) else {}

    pose = poses.get(expression) if expression else None
    if not pose or not isinstance(pose, Mapping):
        pose = poses.get("neutral") or {}

    targets = resolve_pose_targets(pose, servos_map)

    if expression != "neutral":
        if neutral_targets is None:
            neutral_pose = poses.get("neutral") or {}
            neutral_targets = resolve_pose_targets(neutral_pose, servos_map)
        targets = interpolate_expression_targets(targets, neutral_targets, intensity)

    raw_step = pose.get("default_step") if isinstance(pose, Mapping) else None
    fallback_step = default_step if default_step is not None else DEFAULT_STEP_SIZE
    step_size = float(raw_step or fallback_step)

    return DialogExpressionPose(targets=targets, step_size=step_size)
