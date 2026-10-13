import importlib
import sys
import types
from unittest.mock import MagicMock

from services.orchestration.dialog_action_plan import DialogActionPlanWorkflow
from services.orchestration.behavior_tree_workflow import NativeBehaviorTreeWorkflow


def _workflow(*, native_available, execute_plan, execute_action):
    def submit_native(turn_id, user_prompt, actions):
        if not native_available():
            return None
        return NativeBehaviorTreeWorkflow(
            authorize=lambda *_args: (True, ""),
            execute_plan=execute_plan,
        ).invoke(turn_id=turn_id, user_prompt=user_prompt, actions=actions)

    return DialogActionPlanWorkflow(
        submit_native=submit_native,
        authorize=lambda _prompt, _name, _arguments: (True, ""),
        execute_action=execute_action,
        cancelled=lambda: False,
    )


def _actions():
    return [{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}]


def test_prefers_native_result_when_native_transport_is_available():
    plans = []
    fallback_calls = []
    workflow = _workflow(
        native_available=lambda: True,
        execute_plan=lambda plan: plans.append(plan) or {
            "plan_id": plan.plan_id,
            "status": "success",
            "results": [{
                "step_id": plan.steps[0].step_id,
                "name": plan.steps[0].name,
                "status": "completed",
            }],
        },
        execute_action=lambda *args: fallback_calls.append(args),
    )

    result = workflow.invoke(actions=_actions(), user_prompt="挥手", turn_id="turn")

    assert result["status"] == "success"
    assert len(plans) == 1
    assert fallback_calls == []


def test_falls_back_to_correlated_actions_when_native_transport_is_absent():
    calls = []
    workflow = _workflow(
        native_available=lambda: False,
        execute_plan=lambda _plan: (_ for _ in ()).throw(AssertionError("not native")),
        execute_action=lambda name, arguments, turn_id: calls.append(
            (name, arguments, turn_id)
        ) or {"status": "completed", "action": name},
    )

    result = workflow.invoke(actions=_actions(), user_prompt="挥手", turn_id="turn")

    assert result["status"] == "success"
    assert calls == [("play_sequence", {"sequence_name": "wave_hello"}, "turn")]


def test_falls_back_when_native_transport_exists_but_owner_is_unavailable():
    calls = []
    workflow = _workflow(
        native_available=lambda: True,
        execute_plan=lambda _plan: None,
        execute_action=lambda name, arguments, turn_id: calls.append(
            (name, arguments, turn_id)
        ) or {"status": "completed", "action": name},
    )

    result = workflow.invoke(actions=_actions(), user_prompt="挥手", turn_id="turn")

    assert result["status"] == "success"
    assert calls == [("play_sequence", {"sequence_name": "wave_hello"}, "turn")]


def test_native_terminal_result_never_reexecutes_actions():
    for status in ("failure", "halted", "rejected"):
        native_result = {"status": status, "error": "native_execution_stopped"}
        fallback_calls = []
        workflow = DialogActionPlanWorkflow(
            submit_native=lambda *_args: native_result,
            authorize=lambda *_args: (True, ""),
            execute_action=lambda *args: fallback_calls.append(args),
            cancelled=lambda: False,
        )

        result = workflow.invoke(actions=_actions(), user_prompt="挥手", turn_id="turn")

        assert result is native_result
        assert fallback_calls == []


def test_unavailable_native_path_preserves_caller_authorization_policy():
    authorization_calls = []
    execution_calls = []

    def authorize(prompt, name, arguments):
        authorization_calls.append((prompt, name, arguments))
        return False, "non_command_context"

    workflow = DialogActionPlanWorkflow(
        submit_native=lambda *_args: None,
        authorize=authorize,
        execute_action=lambda *args: execution_calls.append(args),
        cancelled=lambda: False,
    )

    result = workflow.invoke(
        actions=_actions(), user_prompt="你会挥手吗", turn_id="question-turn"
    )

    assert authorization_calls == [
        ("你会挥手吗", "play_sequence", {"sequence_name": "wave_hello"})
    ]
    assert result["results"][0]["reason"] == "non_command_context"
    assert execution_calls == []


def test_cancelled_fallback_does_not_dispatch_actions():
    execution_calls = []
    workflow = DialogActionPlanWorkflow(
        submit_native=lambda *_args: None,
        authorize=lambda *_args: (True, ""),
        execute_action=lambda *args: execution_calls.append(args),
        cancelled=lambda: True,
    )

    result = workflow.invoke(actions=_actions(), user_prompt="挥手", turn_id="turn")

    assert result["status"] == "halted"
    assert result["results"][0]["reason"] == "turn_cancelled"
    assert execution_calls == []


def _text_node():
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object
    std_msgs = types.ModuleType("std_msgs")
    messages = types.ModuleType("std_msgs.msg")
    messages.String = messages.UInt8MultiArray = object
    overrides = {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "std_msgs": std_msgs,
        "std_msgs.msg": messages,
    }
    saved_modules = {
        name: sys.modules[name] for name in overrides if name in sys.modules
    }
    sys.modules.update(overrides)
    try:
        module = importlib.import_module("nodes.llm_ros_node")
    finally:
        # Keep newly imported dependencies cached: their import hooks can
        # outlive a wholesale sys.modules rollback (e.g. beartype on the board).
        for name in overrides:
            if name in saved_modules:
                sys.modules[name] = saved_modules[name]
            else:
                sys.modules.pop(name, None)
        sys.modules.pop("nodes.llm_ros_node", None)
    node = module.LLMBrainNode.__new__(module.LLMBrainNode)
    node._behavior_tree_executor = object()
    node._voice_task_cancelled = MagicMock(return_value=False)
    node._execute_dialog_action = MagicMock()
    node._try_execute_native_plan = MagicMock()
    return node


def test_text_entry_preserves_native_plan_identity_and_cancellation():
    node = _text_node()

    def complete_plan(plan, *, cancelled):
        assert plan.turn_id == "text-turn"
        assert plan.user_prompt == "挥手"
        assert not cancelled()
        node._voice_task_cancelled.return_value = True
        assert cancelled()
        return {
            "plan_id": plan.plan_id,
            "status": "success",
            "results": [{
                "step_id": plan.steps[0].step_id,
                "name": plan.steps[0].name,
                "status": "completed",
            }],
        }

    node._try_execute_native_plan.side_effect = complete_plan

    result = node._execute_dialog_action_sequence(
        _actions(), user_prompt="挥手", turn_id="text-turn"
    )

    assert result["status"] == "success"
    node._try_execute_native_plan.assert_called_once()
    node._execute_dialog_action.assert_not_called()


def test_text_entry_still_rejects_capability_question_before_native_submission():
    node = _text_node()

    result = node._execute_dialog_action_sequence(
        _actions(), user_prompt="你会挥手吗", turn_id="question-turn"
    )

    assert result["status"] == "failure"
    assert result["results"][0]["status"] == "rejected"
    node._try_execute_native_plan.assert_not_called()
    node._execute_dialog_action.assert_not_called()
