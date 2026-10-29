"""One outer route is shared by request and plan acceptance boundaries."""

import json

import pytest

from services.llm.llm_conditional_planning import (
    evaluate_conditional_plan, evaluate_native_conditional_event,
)
from services.llm.llm_request_preparation import prepare_voice_request
from services.llm.llm_tool_proposal import evaluate_tool_proposal
from services.orchestration.task_routing import select_task_route


@pytest.mark.parametrize("text", [
    "先低头，然后找红盒子，如果找到就挥手",
    "先低头，然后找红盒子，找到后挥手",
    "先低头，前面没人的话就举手",
    "不要低头然后挥手再看",
    "如果前面有人，就先挥手再低头",
    "如果前面有人，就挥手、低头",
    "如果前面有人，就低头看看",
    "如果前面有人，就点头，然后看看前面",
])
def test_independent_steps_require_a_complete_outer_plan(text):
    assert select_task_route(text) == "ordered_task"
    assert prepare_voice_request(text).route == "ordered_task"


@pytest.mark.parametrize("text", [
    "如果看到红盒子，就挥手",
    "看看前面，如果有人低头就挥手",
    "先看看有没有低头的人，如果有就挥手",
    "如果没有看到红盒子，就挥手",
    "如果看到红灯，就帮我停止跟随",
    "如果看到障碍物，请停止跟随",
])
def test_one_shot_condition_preserves_its_original_route(text):
    assert select_task_route(text) == "conditional_task"
    assert prepare_voice_request(text).route == "conditional_task"


@pytest.mark.parametrize("text", ["停止跟随", "不要再看着我了", "不要再看住我"])
def test_standalone_tracking_stop_keeps_its_immediate_path(text):
    request = prepare_voice_request(text)
    assert select_task_route(text) is None
    assert request.route == "safety_action"
    assert request.safety_action_arguments == {"mode": "idle"}


@pytest.mark.parametrize("text", [
    "请复述如果看到人就点头",
    "如果看到障碍物，请问应该怎么办",
    "如果看到障碍物，请问你能停止跟随吗",
])
def test_text_transformations_and_conditional_questions_do_not_become_a_task(text):
    assert select_task_route(text) is None
    assert prepare_voice_request(text).route != "safety_action"


@pytest.mark.parametrize("text", [
    "先低头，然后找红盒子，如果找到就挥手",
    "先低头，然后找红盒子，找到后挥手",
])
def test_caller_hint_and_conditional_fallback_cannot_drop_independent_steps(text):
    plan = {
        "observation": "观察前方有没有红盒子", "condition": "找到红盒子",
        "action_name": "play_sequence", "action_arguments": {"sequence_name": "wave_hello"},
    }
    event = {"type": "tool_call", "name": "run_conditional_task",
             "arguments": json.dumps(plan, ensure_ascii=False)}
    for hint in (False, True):
        decision = evaluate_tool_proposal(event, user_prompt=text, conditional_request=hint)
        assert decision.is_rejected
        assert decision.rejection_reason == "ordered_task_must_stay_atomic"
    assert evaluate_conditional_plan(plan, user_prompt=text).is_rejected
    assert evaluate_native_conditional_event(event, user_prompt=text).is_rejected
