"""Compile model tool calls into a bounded, server-owned action plan.

The model proposes action names and arguments.  The application assigns plan
and step identities, dependencies, resource labels, and failure policy so the
runtime never has to trust model-authored orchestration metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from services.action_registry import get_action_skill


MAX_ACTION_PLAN_STEPS = 8


class PlanValidationError(ValueError):
    """Raised when model-proposed actions cannot form a safe action plan."""


@dataclass(frozen=True)
class ActionPlanStep:
    step_id: str
    name: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...]
    resources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "arguments": dict(self.arguments),
            "depends_on": list(self.depends_on),
            "resources": list(self.resources),
        }


@dataclass(frozen=True)
class ActionPlan:
    plan_id: str
    turn_id: str
    user_prompt: str
    steps: tuple[ActionPlanStep, ...]
    schema_version: int = 1
    root_type: str = "Sequence"
    on_failure: str = "stop_remaining"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "turn_id": self.turn_id,
            "user_prompt": self.user_prompt,
            "root_type": self.root_type,
            "steps": [step.to_dict() for step in self.steps],
            "on_failure": self.on_failure,
        }


def compile_action_plan(
    *,
    turn_id: str,
    user_prompt: str,
    actions: list[dict[str, Any]],
    max_steps: int = MAX_ACTION_PLAN_STEPS,
) -> ActionPlan:
    """Build the phase-one linear Sequence plan from proposed tool calls."""
    if not isinstance(actions, list) or not actions:
        raise PlanValidationError("action_plan_empty")
    if len(actions) > max_steps:
        raise PlanValidationError("action_plan_too_many_steps")

    steps: list[ActionPlanStep] = []
    previous_step_id = ""
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            raise PlanValidationError(f"invalid_action_at_step_{index}")
        name = action.get("name")
        arguments = action.get("arguments", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise PlanValidationError(f"invalid_action_at_step_{index}")

        name = name.strip()
        skill = get_action_skill(name)
        if skill is None:
            raise PlanValidationError(f"unknown_action_at_step_{index}")
        step_id = f"step-{index:02d}"
        steps.append(ActionPlanStep(
            step_id=step_id,
            name=name,
            arguments=dict(arguments),
            depends_on=(previous_step_id,) if previous_step_id else (),
            resources=skill.plan_resources,
        ))
        previous_step_id = step_id

    return ActionPlan(
        plan_id=f"plan-{uuid4().hex}",
        turn_id=str(turn_id or ""),
        user_prompt=str(user_prompt or ""),
        steps=tuple(steps),
    )


__all__ = [
    "ActionPlan",
    "ActionPlanStep",
    "MAX_ACTION_PLAN_STEPS",
    "PlanValidationError",
    "compile_action_plan",
]
