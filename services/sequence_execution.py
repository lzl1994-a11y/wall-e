"""ROS-independent sequence expansion and servo trajectory execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from services.servo_motion_config import resolve_servo_target


ErrorReporter = Callable[[str], None]


def load_yaml_mapping(
    path: Path | str,
    *,
    on_error: ErrorReporter | None = None,
) -> dict[str, Any]:
    """Load a YAML object, returning an empty mapping on deployment errors."""
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        if on_error is not None:
            on_error(f"Load {path} failed: {exc}")
        return {}
    if isinstance(value, dict):
        return value
    if on_error is not None:
        on_error(f"Load {path} failed: YAML root must be a mapping")
    return {}


class SequenceLibrary:
    """Resolve named poses and recursively expand sequences into a timeline."""

    def __init__(
        self,
        sequences: Mapping[str, Any] | None,
        poses: Mapping[str, Any] | None,
        *,
        on_error: ErrorReporter | None = None,
        max_depth: int = 10,
    ) -> None:
        self.sequences = dict(sequences or {})
        self.poses = dict(poses or {})
        self._on_error = on_error
        self._max_depth = max(0, int(max_depth))

    def flatten(
        self,
        sequence_name: str,
        *,
        offset_time: float = 0.0,
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        """Expand nested sequences into timestamped action frames."""
        if depth > self._max_depth:
            if self._on_error is not None:
                self._on_error(
                    f"Sequence max recursion depth exceeded at {sequence_name}"
                )
            return []

        sequence = self.sequences.get(sequence_name)
        if not sequence:
            if sequence_name in self.poses:
                return [{
                    "time": offset_time,
                    "actions": [{"type": "pose", "name": sequence_name}],
                }]
            return []

        # Preserve the legacy mapping form whose first list value is the
        # actual sequence timeline.
        if isinstance(sequence, dict):
            timelines = [value for value in sequence.values() if isinstance(value, list)]
            if not timelines:
                return []
            sequence = timelines[0]

        frames: list[dict[str, Any]] = []
        for item in sequence:
            if not isinstance(item, dict) or "time" not in item:
                continue
            timestamp = item["time"] + offset_time
            actions: list[dict[str, Any]] = []
            for action in item.get("actions", []):
                if not isinstance(action, dict):
                    continue
                if action.get("type") == "sequence":
                    frames.extend(self.flatten(
                        action.get("name"),
                        offset_time=timestamp,
                        depth=depth + 1,
                    ))
                else:
                    actions.append(action)
            if actions:
                frames.append({"time": timestamp, "actions": actions})
        return frames


class ServoTrajectory:
    """Maintain safe servo targets and advance them one interpolation tick."""

    def __init__(self, servos: Mapping[str, Mapping[str, Any]] | None) -> None:
        self.servos = dict(servos or {})
        self.virtual_state: dict[str, float] = {}
        self.targets: dict[str, float] = {}
        self.steps: dict[str, float] = {}
        for name, config in self.servos.items():
            initial = float(config.get("init", 150))
            self.virtual_state[name] = initial
            self.targets[name] = initial
            self.steps[name] = 0.0
        self._first_tick = True

    def clamp(self, name: str, raw_pwm: Any) -> int | None:
        config = self.servos.get(name)
        return resolve_servo_target(config, raw_pwm) if config else None

    def initial(self, name: str, fallback: float) -> float:
        return float(self.servos.get(name, {}).get("init", fallback))

    def apply_targets(self, targets: Any, step_size: Any) -> None:
        if not isinstance(targets, dict):
            return
        try:
            resolved_step = max(1.0, min(float(step_size), 1000.0))
        except (TypeError, ValueError):
            return
        for name, raw_pwm in targets.items():
            if name not in self.servos:
                continue
            target = self.clamp(name, raw_pwm)
            if target is not None:
                self.targets[name] = target
                self.steps[name] = resolved_step

    def reset_to_initial(self, *, step_size: float = 2.0) -> None:
        for name, config in self.servos.items():
            self.targets[name] = float(config["init"])
            self.steps[name] = float(step_size)

    def tick(self) -> dict[str, int]:
        """Advance one control tick and return only changed servo positions."""
        self._apply_mechanical_target_constraints()
        changed = set()
        if self._first_tick:
            self._first_tick = False
            changed.update(self.virtual_state)

        head_center = self.initial("head_yaw", 5000)
        eye_r_initial = self.initial("eye_r", 3000)
        eye_l_initial = self.initial("eye_l", 6500)
        eye_gap = 3000.0

        for name in list(self.virtual_state):
            target = self.targets[name]
            step = self.steps[name]
            current = self.virtual_state[name]
            if step <= 0 or current == target:
                continue

            if abs(target - current) <= step:
                next_value = target
            elif target > current:
                next_value = current + step
            else:
                next_value = current - step

            # Prevent transient collisions while the coupled head and eyes are
            # moving toward an otherwise safe final pose.
            if name == "head_yaw" and next_value > head_center:
                if self.virtual_state.get("eye_r", eye_r_initial) < eye_r_initial:
                    next_value = head_center
            if name == "eye_r" and next_value < eye_r_initial:
                if self.virtual_state.get("head_yaw", head_center) > head_center:
                    next_value = eye_r_initial
            if name == "head_yaw" and next_value < head_center:
                if self.virtual_state.get("eye_l", eye_l_initial) > eye_l_initial:
                    next_value = head_center
            if name == "eye_l" and next_value > eye_l_initial:
                if self.virtual_state.get("head_yaw", head_center) < head_center:
                    next_value = eye_l_initial

            if name == "eye_l":
                eye_r = self.virtual_state.get("eye_r", eye_r_initial)
                minimum = eye_r + eye_gap
                if self.virtual_state.get("head_yaw", head_center) < head_center:
                    next_value = min(next_value, eye_l_initial)
                elif next_value < minimum:
                    next_value = minimum
            if name == "eye_r":
                eye_l = self.virtual_state.get("eye_l", eye_l_initial)
                next_value = min(next_value, eye_l - eye_gap)

            self.virtual_state[name] = next_value
            changed.add(name)

        return {name: int(self.virtual_state[name]) for name in changed}

    def _apply_mechanical_target_constraints(self) -> None:
        head_center = self.initial("head_yaw", 5000)
        eye_r_initial = self.initial("eye_r", 3000)
        eye_l_initial = self.initial("eye_l", 6500)
        eye_gap = 3000.0
        head_target = self.targets.get("head_yaw", head_center)

        if head_target > head_center:
            if self.targets.get("eye_r", eye_r_initial) < eye_r_initial:
                self.targets["eye_r"] = eye_r_initial
                if self.steps.get("eye_r", 0) <= 0:
                    self.steps["eye_r"] = 30.0
        if head_target < head_center:
            if self.targets.get("eye_l", eye_l_initial) > eye_l_initial:
                self.targets["eye_l"] = eye_l_initial
                if self.steps.get("eye_l", 0) <= 0:
                    self.steps["eye_l"] = 30.0

        eye_r_target = self.targets.get("eye_r", eye_r_initial)
        eye_l_target = self.targets.get("eye_l", eye_l_initial)
        if head_target < head_center:
            maximum_r = eye_l_target - eye_gap
            if eye_r_target > maximum_r:
                self.targets["eye_r"] = maximum_r
                if self.steps.get("eye_r", 0) <= 0:
                    self.steps["eye_r"] = 30.0
        else:
            minimum_l = eye_r_target + eye_gap
            if eye_l_target < minimum_l:
                self.targets["eye_l"] = minimum_l
                if self.steps.get("eye_l", 0) <= 0:
                    self.steps["eye_l"] = 30.0

            eye_l_target = self.targets.get("eye_l", eye_l_initial)
            maximum_r = eye_l_target - eye_gap
            if eye_r_target > maximum_r:
                self.targets["eye_r"] = maximum_r
                if self.steps.get("eye_r", 0) <= 0:
                    self.steps["eye_r"] = 30.0
