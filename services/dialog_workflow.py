"""Deterministic dialog workflows built on LangGraph.

The graph owns orchestration only. ROS, camera, and model access stay behind
injected callbacks so changing this layer cannot bypass the hardware safety
boundary or couple the workflow to a particular LLM provider.
"""

from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from services.conditional_task import (
    ConditionalDecision,
    ConditionalTaskPlan,
    normalize_conditional_task_plan,
    parse_conditional_decision,
)


class CameraInspectionState(TypedDict, total=False):
    turn_id: str
    user_prompt: str
    frame: bytes
    answer: str
    error: str


class CameraInspectionWorkflow:
    """Capture one frame, analyze it, and return one user-facing answer."""

    def __init__(
        self,
        *,
        capture: Callable[[], Any],
        analyze: Callable[[bytes, str], str],
    ):
        self._capture = capture
        self._analyze = analyze

        builder = StateGraph(CameraInspectionState)
        builder.add_node("capture_camera", self._capture_camera)
        builder.add_node("analyze_image", self._analyze_image)
        builder.add_edge(START, "capture_camera")
        builder.add_conditional_edges(
            "capture_camera",
            self._route_after_capture,
            {"analyze": "analyze_image", "finish": END},
        )
        builder.add_edge("analyze_image", END)
        self._graph = builder.compile()

    def invoke(self, *, turn_id: str, user_prompt: str) -> CameraInspectionState:
        return self._graph.invoke({
            "turn_id": turn_id,
            "user_prompt": user_prompt,
        })

    def _capture_camera(self, _state: CameraInspectionState) -> CameraInspectionState:
        preview = self._capture()
        if getattr(preview, "busy", False):
            return {
                "answer": "我正在处理上一张画面，等一下再看。",
                "error": "camera_preview_busy",
            }
        frame = getattr(preview, "last_frame", None)
        if not frame:
            return {
                "answer": "我现在看不到画面，检查一下摄像头连接。",
                "error": getattr(preview, "error", None) or "camera_frame_unavailable",
            }
        return {"frame": frame}

    @staticmethod
    def _route_after_capture(
        state: CameraInspectionState,
    ) -> Literal["analyze", "finish"]:
        return "analyze" if state.get("frame") else "finish"

    def _analyze_image(self, state: CameraInspectionState) -> CameraInspectionState:
        try:
            answer = self._analyze(state["frame"], state["user_prompt"])
            if not answer:
                raise RuntimeError("视觉模型返回空答案")
            return {"answer": answer}
        except Exception as exc:
            return {
                "answer": "这张图我没分析出来，你换个角度再让我看看。",
                "error": str(exc),
            }


class ConditionalTaskState(TypedDict, total=False):
    turn_id: str
    user_prompt: str
    plan: ConditionalTaskPlan
    frame: bytes
    decision: Literal["yes", "no", "uncertain"]
    evidence: str
    action_result: dict[str, Any]
    answer: str
    error: str


class ConditionalTaskWorkflow:
    """Observe once, evaluate one condition, and execute at most one action."""

    def __init__(
        self,
        *,
        capture: Callable[[], Any],
        evaluate: Callable[[bytes, str, str], ConditionalDecision | dict | str],
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute: Callable[[str, dict[str, Any]], dict[str, Any]],
    ):
        self._capture = capture
        self._evaluate = evaluate
        self._authorize = authorize
        self._execute = execute

        builder = StateGraph(ConditionalTaskState)
        builder.add_node("capture_camera", self._capture_camera)
        builder.add_node("evaluate_condition", self._evaluate_condition)
        builder.add_node("authorize_action", self._authorize_action)
        builder.add_node("execute_action", self._execute_action)
        builder.add_node("finish_without_action", self._finish_without_action)
        builder.add_edge(START, "capture_camera")
        builder.add_conditional_edges(
            "capture_camera",
            lambda state: "evaluate" if state.get("frame") else "finish",
            {"evaluate": "evaluate_condition", "finish": END},
        )
        builder.add_conditional_edges(
            "evaluate_condition",
            lambda state: "authorize" if state.get("decision") == "yes" else "finish",
            {"authorize": "authorize_action", "finish": "finish_without_action"},
        )
        builder.add_conditional_edges(
            "authorize_action",
            lambda state: "execute" if not state.get("error") else "finish",
            {"execute": "execute_action", "finish": END},
        )
        builder.add_edge("execute_action", END)
        builder.add_edge("finish_without_action", END)
        self._graph = builder.compile()

    def invoke(
        self,
        *,
        turn_id: str,
        user_prompt: str,
        plan: dict[str, Any],
    ) -> ConditionalTaskState:
        normalized = normalize_conditional_task_plan(plan)
        return self._graph.invoke({
            "turn_id": turn_id,
            "user_prompt": user_prompt,
            "plan": normalized,
        })

    def _capture_camera(self, _state: ConditionalTaskState) -> ConditionalTaskState:
        preview = self._capture()
        if getattr(preview, "busy", False):
            return {
                "answer": "我正在处理上一张画面，这次任务没有执行。",
                "error": "camera_preview_busy",
            }
        frame = getattr(preview, "last_frame", None)
        if not frame:
            return {
                "answer": "我现在看不到画面，所以没有执行动作。",
                "error": getattr(preview, "error", None) or "camera_frame_unavailable",
            }
        return {"frame": frame}

    def _evaluate_condition(self, state: ConditionalTaskState) -> ConditionalTaskState:
        plan = state["plan"]
        try:
            raw = self._evaluate(
                state["frame"],
                plan["observation"],
                plan["condition"],
            )
            decision = parse_conditional_decision(raw)
            return dict(decision)
        except Exception as exc:
            return {
                "decision": "uncertain",
                "evidence": "condition_evaluation_failed",
                "error": str(exc),
            }

    def _authorize_action(self, state: ConditionalTaskState) -> ConditionalTaskState:
        plan = state["plan"]
        allowed, reason = self._authorize(
            state["user_prompt"],
            plan["action_name"],
            plan["action_arguments"],
        )
        if allowed:
            return {}
        return {
            "answer": "条件满足，但这个动作没有通过安全检查。",
            "error": reason or "action_not_authorized",
        }

    def _execute_action(self, state: ConditionalTaskState) -> ConditionalTaskState:
        plan = state["plan"]
        try:
            result = self._execute(plan["action_name"], plan["action_arguments"])
        except Exception as exc:
            result = {
                "status": "failed",
                "action": plan["action_name"],
                "reason": str(exc),
            }
        status = result.get("status") if isinstance(result, dict) else "failed"
        if status == "completed":
            answer = "条件满足，动作已经执行完成。"
            error = ""
        else:
            answer = "条件满足，但动作没有执行成功。"
            error = str(result.get("reason") or status or "action_failed")
        update: ConditionalTaskState = {"action_result": result, "answer": answer}
        if error:
            update["error"] = error
        return update

    @staticmethod
    def _finish_without_action(state: ConditionalTaskState) -> ConditionalTaskState:
        if state.get("decision") == "no":
            return {"answer": "条件不满足，我没有执行动作。"}
        return {"answer": "我没法确定条件是否满足，所以没有执行动作。"}
