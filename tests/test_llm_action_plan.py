from services.llm_action_plan import LLMActionPlanWorkflow


def _workflow(*, native_available, execute_plan, execute_action):
    return LLMActionPlanWorkflow(
        native_available=native_available,
        authorize=lambda _prompt, _name, _arguments: (True, ""),
        execute_plan=execute_plan,
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
