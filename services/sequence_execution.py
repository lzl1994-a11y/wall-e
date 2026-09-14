"""ROS-independent sequence expansion and servo trajectory execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from services.servo_motion_config import resolve_servo_target


ErrorReporter = Callable[[str], None]


DEFAULT_MOTION_TO_MOTOR = {
    "forward": {
        "left": {"action": 1, "throttle": 55},
        "right": {"action": 1, "throttle": 55},
    },
    "backward": {
        "left": {"action": 2, "throttle": 55},
        "right": {"action": 2, "throttle": 55},
    },
    "spin": {
        "left": {"action": 2, "throttle": 55},
        "right": {"action": 1, "throttle": 55},
    },
    "left": {
        "left": {"action": 2, "throttle": 45},
        "right": {"action": 1, "throttle": 55},
    },
    "right": {
        "left": {"action": 1, "throttle": 55},
        "right": {"action": 2, "throttle": 45},
    },
}


@dataclass(frozen=True)
class SequenceEffect:
    """An output requested by the pure sequence runtime."""

    kind: str
    payload: Any = None


@dataclass(frozen=True)
class SequenceTick:
    """Outputs produced by one runtime control tick."""

    effects: tuple[SequenceEffect, ...]
    servo_positions: dict[str, int]


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


class SequenceRuntime:
    """Own the timeline, action dispatch, and timed motor command lifecycle."""

    def __init__(
        self,
        library: SequenceLibrary,
        trajectory: ServoTrajectory,
        *,
        motion_to_motor: Mapping[str, Any] | None = None,
    ) -> None:
        self.library = library
        self.trajectory = trajectory
        self.motion_to_motor = dict(motion_to_motor or DEFAULT_MOTION_TO_MOTOR)
        self.timeline: list[dict[str, Any]] = []
        self.sequence_started_at = 0.0
        self.active_motor_command: Any = None
        self.motor_stop_at = 0.0
        self.explicit_motion_active = False

    def start_sequence(self, name: str, *, now: float) -> int:
        frames = self.library.flatten(name, offset_time=0.0)
        if not frames:
            self.explicit_motion_active = False
            return 0
        frames.sort(key=lambda item: item["time"])
        self.timeline = frames
        self.sequence_started_at = now
        self.explicit_motion_active = True
        return len(frames)

    def clear_sequence(self, *, clear_explicit_motion: bool = False) -> None:
        self.timeline = []
        if clear_explicit_motion:
            self.explicit_motion_active = False

    def halt_interpolation(self) -> None:
        for name in self.trajectory.steps:
            self.trajectory.steps[name] = 0.0

    def stop_motor(self) -> None:
        self.active_motor_command = None
        self.motor_stop_at = 0.0

    def dispatch_action(
        self,
        action: Mapping[str, Any],
        *,
        monotonic_now: float,
    ) -> tuple[SequenceEffect, ...]:
        """Apply an action and return hardware-facing effects for the node."""
        action_type = action.get("type")
        if action_type == "servo":
            name = action.get("name")
            if name in self.trajectory.servos:
                raw_pwm = action.get("pwm", action.get("angle", 4000))
                target = self.trajectory.clamp(name, raw_pwm)
                if target is not None:
                    self.trajectory.targets[name] = target
                    self.trajectory.steps[name] = float(action.get("step_size", 40.0))
            return ()

        if action_type == "pose":
            pose = self.library.poses.get(action.get("name"))
            if pose:
                override_step = action.get("step_size")
                default_step = pose.get("default_step", 2.0)
                step = float(override_step if override_step is not None else default_step)
                for name, raw_pwm in pose.get("targets", {}).items():
                    if name not in self.trajectory.servos:
                        continue
                    target = self.trajectory.clamp(name, raw_pwm)
                    if target is not None:
                        self.trajectory.targets[name] = target
                        self.trajectory.steps[name] = step
            return ()

        if action_type == "motor":
            direction = action.get("direction", "forward")
            duration = max(0.0, min(float(action.get("duration", 1.0)), 10.0))
            command = self.motion_to_motor.get(direction)
            if command is None:
                return ()
            if duration <= 0.0:
                self.stop_motor()
                return (SequenceEffect("motor_stop"),)
            self.active_motor_command = command
            self.motor_stop_at = monotonic_now + duration
            return (SequenceEffect("motor", command),)

        if action_type == "express_emotion":
            return (SequenceEffect("emotion", action.get("emotion", "happy")),)

        if action_type == "manual_servo":
            self.trajectory.apply_targets(
                action.get("targets", {}),
                action.get("step_size", 30.0),
            )
        return ()

    def tick(self, *, wall_now: float, monotonic_now: float) -> SequenceTick:
        effects: list[SequenceEffect] = []
        if self.active_motor_command is not None:
            if monotonic_now >= self.motor_stop_at:
                self.stop_motor()
                effects.append(SequenceEffect("motor_stop"))
            else:
                effects.append(SequenceEffect("motor", self.active_motor_command))

        # Preserve the existing behavior of dispatching at most one timeline
        # frame per 50 Hz tick, even when several frames are already due.
        if self.timeline:
            frame = self.timeline[0]
            if wall_now - self.sequence_started_at >= frame.get("time", 0):
                self.timeline.pop(0)
                for action in frame.get("actions", []):
                    effects.extend(self.dispatch_action(
                        action,
                        monotonic_now=monotonic_now,
                    ))

        return SequenceTick(
            effects=tuple(effects),
            servo_positions=self.trajectory.tick(),
        )

    def motion_complete(self) -> bool:
        return (
            not self.timeline
            and self.active_motor_command is None
            and all(
                self.trajectory.steps.get(name, 0.0) <= 0.0
                or self.trajectory.virtual_state.get(name)
                == self.trajectory.targets.get(name)
                for name in self.trajectory.virtual_state
            )
        )


class SequenceCommandController:
    """Apply high-level commands and own their ROS-independent lifecycle."""

    SUPPORTED_ACTIONS = frozenset({
        "express_emotion",
        "move_chassis",
        "manual_servo",
        "play_sequence",
        "stop_all",
    })

    def __init__(self, runtime: SequenceRuntime) -> None:
        self.runtime = runtime
        self.motor_request: dict[str, Any] | None = None
        self.sequence_request: dict[str, Any] | None = None
        self.pending_dialog_expression: tuple[Any, Any] | None = None
        self.game_active = False

    def handle_action(
        self,
        request: Mapping[str, Any],
        *,
        wall_now: float,
        monotonic_now: float,
    ) -> tuple[SequenceEffect, ...]:
        if self.game_active or request.get("name") not in self.SUPPORTED_ACTIONS:
            return ()
        request = dict(request)
        name = request["name"]
        arguments = request.get("arguments", {})
        effects: list[SequenceEffect] = []

        effects.extend(self._interrupt_sequence("superseded_by_new_command"))
        if self.runtime.active_motor_command is not None:
            effects.extend(self._stop_motor(
                status="interrupted",
                detail="superseded_by_new_command",
            ))
        self.runtime.clear_sequence()
        self.runtime.halt_interpolation()
        effects.append(SequenceEffect("cancel_auto_reset_timer"))
        effects.append(SequenceEffect("log_info", f"[Interrupt] Cleared state for tool: {name}"))

        if name == "express_emotion":
            self._append_status(effects, request, "accepted")
            effects.extend(self.runtime.dispatch_action(
                {"type": "express_emotion", "emotion": arguments.get("emotion", "happy")},
                monotonic_now=monotonic_now,
            ))
            self._append_status(effects, request, "completed")
        elif name == "move_chassis":
            direction = arguments.get("direction", "")
            if direction not in self.runtime.motion_to_motor:
                self._append_status(effects, request, "rejected", "invalid_direction")
                return tuple(effects)
            self.motor_request = request if request.get("request_id") else None
            self._append_status(effects, request, "accepted")
            effects.extend(self._consume_runtime_effects(
                self.runtime.dispatch_action(
                    {
                        "type": "motor",
                        "direction": direction,
                        "duration": float(arguments.get("duration", 1.0)),
                    },
                    monotonic_now=monotonic_now,
                ),
                motor_status="completed",
            ))
        elif name == "manual_servo":
            self.runtime.explicit_motion_active = True
            self._append_status(effects, request, "accepted")
            effects.extend(self.runtime.dispatch_action(
                {
                    "type": "manual_servo",
                    "targets": arguments.get("targets", {}),
                    "step_size": arguments.get("step_size", 30.0),
                },
                monotonic_now=monotonic_now,
            ))
            self._append_status(effects, request, "completed")
        elif name == "play_sequence":
            sequence_name = arguments.get("sequence_name", "")
            frame_count = self.runtime.start_sequence(sequence_name, now=wall_now)
            if frame_count:
                self.sequence_request = request if request.get("request_id") else None
                self._append_status(effects, request, "accepted")
                effects.append(SequenceEffect(
                    "log_info",
                    f"[Sequence] Playing sequence: {sequence_name} ({frame_count} frames)",
                ))
            else:
                effects.append(SequenceEffect(
                    "log_warning",
                    f"[Sequence] Sequence '{sequence_name}' not found or empty",
                ))
                self._append_status(
                    effects, request, "rejected", "unknown_or_empty_sequence"
                )
        elif name == "stop_all":
            effects.extend(self._stop_motor(status="interrupted", detail="stop_all"))
            self._append_status(effects, request, "completed")
        return tuple(effects)

    def cancel(self, cancellation: Mapping[str, Any]) -> tuple[SequenceEffect, ...]:
        request_id = cancellation.get("request_id")
        reason = str(cancellation.get("reason") or "cancelled")
        effects: list[SequenceEffect] = []
        if self.sequence_request is not None and self.sequence_request.get("request_id") == request_id:
            effects.extend(self._interrupt_sequence(reason))
            self.runtime.clear_sequence(clear_explicit_motion=True)
            self.runtime.halt_interpolation()
            effects.append(SequenceEffect("cancel_auto_reset_timer"))
            if self.runtime.active_motor_command is not None and self.motor_request is None:
                effects.extend(self._stop_motor(status="interrupted", detail=reason))
        if self.motor_request is not None and self.motor_request.get("request_id") == request_id:
            effects.extend(self._stop_motor(status="interrupted", detail=reason))
        return tuple(effects)

    def apply_tracking_targets(self, targets: Any, step_size: Any) -> None:
        if not self.game_active:
            self.runtime.trajectory.apply_targets(targets, step_size)

    def apply_dialog_expression(self, targets: Any, step_size: Any) -> None:
        pending = (targets, step_size)
        if (
            self.game_active
            or self.runtime.explicit_motion_active
            or self.runtime.active_motor_command is not None
        ):
            self.pending_dialog_expression = pending
            return
        self.pending_dialog_expression = None
        self.runtime.trajectory.apply_targets(*pending)

    def set_game_active(self, active: bool) -> tuple[SequenceEffect, ...]:
        effects: list[SequenceEffect] = []
        if active and not self.game_active:
            effects.extend(self._interrupt_sequence("game_mode"))
            self.runtime.clear_sequence()
            self.runtime.halt_interpolation()
            effects.append(SequenceEffect("cancel_auto_reset_timer"))
            effects.extend(self._stop_motor(status="interrupted", detail="game_mode"))
        self.game_active = active
        return tuple(effects)

    def tick(self, *, wall_now: float, monotonic_now: float) -> SequenceTick:
        if self.game_active:
            return SequenceTick((), {})
        result = self.runtime.tick(wall_now=wall_now, monotonic_now=monotonic_now)
        effects = list(self._consume_runtime_effects(
            result.effects,
            motor_status="completed",
        ))
        if self.sequence_request is not None and self.runtime.motion_complete():
            request = self.sequence_request
            self.sequence_request = None
            self._append_status(effects, request, "completed")
        if self.runtime.explicit_motion_active and self.runtime.motion_complete():
            self.runtime.explicit_motion_active = False
            if self.pending_dialog_expression is not None:
                pending = self.pending_dialog_expression
                self.pending_dialog_expression = None
                self.runtime.trajectory.apply_targets(*pending)
        return SequenceTick(tuple(effects), result.servo_positions)

    def _interrupt_sequence(self, detail: str) -> tuple[SequenceEffect, ...]:
        request = self.sequence_request
        self.sequence_request = None
        effects: list[SequenceEffect] = []
        self._append_status(effects, request, "interrupted", detail)
        return tuple(effects)

    def _stop_motor(self, *, status: str, detail: str) -> tuple[SequenceEffect, ...]:
        self.runtime.stop_motor()
        request = self.motor_request
        self.motor_request = None
        effects: list[SequenceEffect] = [SequenceEffect("motor_stop")]
        self._append_status(effects, request, status, detail)
        return tuple(effects)

    def _consume_runtime_effects(
        self,
        effects: tuple[SequenceEffect, ...],
        *,
        motor_status: str,
    ) -> tuple[SequenceEffect, ...]:
        consumed: list[SequenceEffect] = []
        for effect in effects:
            if effect.kind == "motor_stop":
                consumed.extend(self._stop_motor(status=motor_status, detail=""))
            else:
                consumed.append(effect)
        return tuple(consumed)

    @staticmethod
    def _append_status(
        effects: list[SequenceEffect],
        request: Mapping[str, Any] | None,
        status: str,
        detail: str = "",
    ) -> None:
        if request is not None and request.get("request_id"):
            effects.append(SequenceEffect("status", {
                "request": dict(request),
                "status": status,
                "detail": detail,
            }))
