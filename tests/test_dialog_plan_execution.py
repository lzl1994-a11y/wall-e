from services.dialog_plan_execution import DialogNativePlanWorkflow


def _workflow(executed):
    return DialogNativePlanWorkflow(
        authorize=lambda _prompt, _name, _arguments: (True, ""),
        execute_plan=lambda plan: executed.append(plan) or {
            "plan_id": plan.plan_id,
            "results": [{
                "step_id": plan.steps[0].step_id,
                "name": plan.steps[0].name,
                "status": "completed",
            }],
            "status": "success",
        },
    )


def test_special_dialog_tools_are_left_for_their_dedicated_workflows():
    executed = []
    result = _workflow(executed).invoke(
        turn_id="turn", user_prompt="看看前面",
        actions=[{"name": "inspect_camera", "arguments": {"question": "前面有什么"}}],
    )

    assert result is None
    assert executed == []


def test_ordinary_dialog_actions_are_submitted_to_the_native_workflow():
    executed = []
    result = _workflow(executed).invoke(
        turn_id="turn", user_prompt="挥手",
        actions=[{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}],
    )

    assert result["status"] == "success"
    assert len(executed) == 1
