"""Bounded semantic repair of ordered plans, before any execution begins."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from services.orchestration.ordered_task import validate_ordered_task_plan


class _PlanningState(TypedDict, total=False):
    user_prompt: str
    candidate: Any
    plan: dict[str, Any] | None
    error: str | None
    attempts: int
    stopped: bool


class OrderedTaskPlanningWorkflow:
    """Validate each candidate; at most one repair, no transport retries.

    ``propose`` receives empty feedback initially. A repair receives JSON with
    the rejected candidate and exact validator error. The caller retains the
    original user request when constructing every model call.
    """

    def __init__(self, *, propose: Callable[[str], Any],
                 cancelled: Callable[[], bool] | None = None):
        self._propose = propose
        self._cancelled = cancelled or (lambda: False)
        builder = StateGraph(_PlanningState)
        builder.add_node("propose", self._propose_candidate)
        builder.add_node("validate", self._validate_candidate)
        builder.add_conditional_edges(
            START, lambda state: "validate" if "candidate" in state else "propose",
            {"validate": "validate", "propose": "propose"},
        )
        builder.add_conditional_edges(
            "propose", lambda state: "finish" if state.get("stopped") else "validate",
            {"finish": END, "validate": "validate"},
        )
        builder.add_conditional_edges(
            "validate", lambda state: (
                "repair" if state.get("error") and not state.get("stopped")
                and state["attempts"] < 2 else "finish"
            ), {"repair": "propose", "finish": END},
        )
        self._graph = builder.compile()

    def invoke(self, user_prompt: str, initial_candidate: Any = None) -> dict[str, Any]:
        if self._cancelled():
            return {"plan": None, "error": "turn_cancelled", "attempts": 0}
        state: _PlanningState = {
            "user_prompt": user_prompt, "plan": None, "error": None,
            "attempts": 0, "stopped": False,
        }
        if initial_candidate is not None:
            state.update(candidate=initial_candidate, attempts=1)
        result = self._graph.invoke(state)
        return {key: result.get(key) for key in ("plan", "error", "attempts")}

    def _propose_candidate(self, state: _PlanningState) -> dict[str, Any]:
        if self._cancelled():
            return {"plan": None, "error": "turn_cancelled", "stopped": True}
        attempts = state["attempts"] + 1
        try:
            feedback = ""
            if state.get("error"):
                feedback = json.dumps({
                    "validation_error": state["error"], "candidate": state.get("candidate"),
                }, ensure_ascii=False, default=str)
            candidate = self._propose(feedback)
        except Exception as exc:
            return {"plan": None, "error": f"ordered_planning_failed:{type(exc).__name__}",
                    "attempts": attempts, "stopped": True}
        return {"candidate": candidate, "attempts": attempts}

    def _validate_candidate(self, state: _PlanningState) -> dict[str, Any]:
        if self._cancelled():
            return {"plan": None, "error": "turn_cancelled", "stopped": True}
        try:
            candidate = state.get("candidate")
            if isinstance(candidate, str):
                candidate = json.loads(candidate)
            plan = validate_ordered_task_plan(state["user_prompt"], candidate)
        except json.JSONDecodeError as exc:
            return {"plan": None, "error": f"ordered_plan_json_invalid:{exc.msg}"}
        except (TypeError, ValueError) as exc:
            return {"plan": None, "error": str(exc)}
        if self._cancelled():
            return {"plan": None, "error": "turn_cancelled", "stopped": True}
        return {"plan": plan, "error": None}
