"""Unit tests for services.llm_conditional_planning."""

from __future__ import annotations

import json
import sys
import pytest

from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.llm_conditional_planning import (
    ConditionalFallbackRequest,
    ConditionalPlanDecision,
    LLMConditionalPlanning,
    STATUS_ACCEPTED,
    STATUS_IGNORED,
    STATUS_MALFORMED,
    STATUS_REJECTED,
    build_conditional_fallback_prompt,
    build_conditional_fallback_request,
    evaluate_conditional_plan,
    evaluate_native_conditional_event,
    parse_conditional_fallback_json,
)


def test_ordinary_text_event_ignored() -> None:
    event = {"type": "text", "content": "收到，马上为您观察"}
    decision = evaluate_native_conditional_event(event, user_prompt="如果看到猫就招手")
    assert decision.is_ignored
    assert not decision.is_accepted
    assert not decision.is_rejected
    assert not decision.is_malformed
    assert decision.status == STATUS_IGNORED
    assert decision.plan is None


def test_other_tool_call_ignored() -> None:
    event = {
        "type": "tool_call",
        "name": "move_chassis",
        "arguments": json.dumps({"direction": "forward", "duration": 1}),
    }
    decision = evaluate_native_conditional_event(event, user_prompt="如果看到猫就招手")
    assert decision.is_ignored
    assert decision.plan is None


def test_valid_conditional_tool_call_accepted() -> None:
    raw_arguments = {
        "observation": "检查前方是否有猫",
        "condition": "画面中出现猫",
        "action_name": "play_sequence",
        "action_arguments": {"sequence_name": "wave_hello"},
    }
    event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps(raw_arguments, ensure_ascii=False),
    }
    decision = evaluate_native_conditional_event(event, user_prompt="如果看到猫就招手")
    assert decision.is_accepted
    assert decision.status == STATUS_ACCEPTED
    assert decision.plan is not None
    assert decision.plan["action_name"] == "play_sequence"
    assert decision.plan["action_arguments"] == {"sequence_name": "wave_hello"}
    assert decision.rejection_reason is None


def test_malformed_json_returns_malformed() -> None:
    event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": '{"observation": "看到猫", "condition": "看到猫", "action_name": ',
    }
    decision = evaluate_native_conditional_event(event, user_prompt="如果看到猫就招手")
    assert decision.is_malformed
    assert decision.status == STATUS_MALFORMED
    assert not decision.is_accepted
    assert decision.plan is None

    non_string_event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": {"observation": "看到猫"},
    }
    non_string_decision = evaluate_native_conditional_event(
        non_string_event, user_prompt="如果看到猫就招手"
    )
    assert non_string_decision.is_malformed


def test_json_non_object_retains_arguments_not_object() -> None:
    array_event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps(["observation", "condition"]),
    }
    decision_array = evaluate_native_conditional_event(
        array_event, user_prompt="如果看到猫就招手"
    )
    assert decision_array.is_rejected
    assert decision_array.status == STATUS_REJECTED
    assert decision_array.rejection_reason == "arguments_not_object"

    string_event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps("just a string"),
    }
    decision_string = evaluate_native_conditional_event(
        string_event, user_prompt="如果看到猫就招手"
    )
    assert decision_string.is_rejected
    assert decision_string.rejection_reason == "arguments_not_object"


def test_canonicalize_conditional_action_corrects_explicit_action() -> None:
    raw_arguments = {
        "observation": "检查前方是否有猫",
        "condition": "画面中出现猫",
        "action_name": "play_sequence",
        "action_arguments": {"sequence_name": "wave_hello"},
    }
    event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps(raw_arguments, ensure_ascii=False),
    }
    # User explicitly asked to "点头"
    decision = evaluate_native_conditional_event(
        event, user_prompt="看看前面，如果看到猫你就点头"
    )
    assert decision.is_accepted
    assert decision.plan is not None
    assert decision.plan["action_name"] == "play_sequence"
    assert decision.plan["action_arguments"] == {"sequence_name": "basic_nod"}


def test_validate_action_call_rejection_reason_preserved() -> None:
    # Model proposes disallowed action name not in CONDITIONAL_ACTION_TOOLS
    raw_arguments = {
        "observation": "检查前方是否有猫",
        "condition": "画面中出现猫",
        "action_name": "disallowed_custom_action",
        "action_arguments": {},
    }
    event = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps(raw_arguments, ensure_ascii=False),
    }
    decision = evaluate_native_conditional_event(
        event, user_prompt="如果看到猫就做动作"
    )
    assert decision.is_rejected
    assert decision.rejection_reason == "invalid_arguments"

    # Also test non-conditional command prompt rejection
    valid_arguments = {
        "observation": "检查前方是否有猫",
        "condition": "画面中出现猫",
        "action_name": "play_sequence",
        "action_arguments": {"sequence_name": "basic_nod"},
    }
    event_valid = {
        "type": "tool_call",
        "name": CONDITIONAL_TASK_TOOL_NAME,
        "arguments": json.dumps(valid_arguments, ensure_ascii=False),
    }
    decision_non_cond = evaluate_native_conditional_event(
        event_valid, user_prompt="今天天气真好"
    )
    assert decision_non_cond.is_rejected
    assert decision_non_cond.rejection_reason == "not_conditional_command"


def test_accepted_decision_carries_normalized_plan() -> None:
    raw_plan = {
        "observation": "有人举手",
        "condition": "看到有人举手",
        "action_name": "express_emotion",
        "action_arguments": {"emotion": "happy"},
    }
    decision = evaluate_conditional_plan(
        raw_plan, user_prompt="如果看到有人举手就笑一个"
    )
    assert decision.is_accepted
    assert decision.plan == {
        "observation": "有人举手",
        "condition": "看到有人举手",
        "action_name": "express_emotion",
        "action_arguments": {"emotion": "happy"},
    }


def test_fallback_prompt_matches_node_exactly() -> None:
    user_prompt = "看到红苹果就举手"
    prompt = build_conditional_fallback_prompt(user_prompt)
    expected = (
        "把下面的现实条件任务转换成一个 JSON 对象。只能输出 JSON，不要回答任务，"
        "不要声称已经观察或执行。对象必须且只能包含 observation、condition、"
        "action_name、action_arguments。condition 保留用户完整的肯定或否定条件。"
        "action_name 只能是 play_sequence、express_emotion、set_tracking_mode、"
        "set_vision_gate、stop_all。常用 play_sequence 参数：举手=raise_hand，"
        "点头=basic_nod，挥手=wave_hello，双手放下=arms_down，回正=look_center；"
        "action_arguments 必须是对应动作的参数对象。禁止使用 move_chassis。\n"
        f"用户任务：{user_prompt}"
    )
    assert prompt == expected


def test_fallback_request_configuration() -> None:
    req = build_conditional_fallback_request("如果看到猫就点头")
    assert isinstance(req, ConditionalFallbackRequest)
    assert req.tools_enabled is False
    assert req.structured_answer is False
    assert req.max_tokens_override == 384
    assert req.history == []
    assert req.system_prompt == "你是机器人条件任务规划器，只做受限 JSON 转换，不观察环境、不执行动作。"


def test_parse_plain_json() -> None:
    raw = (
        '{"observation": "看到猫", "condition": "看到猫", '
        '"action_name": "play_sequence", "action_arguments": {"sequence_name": "basic_nod"}}'
    )
    parsed = parse_conditional_fallback_json(raw)
    assert isinstance(parsed, dict)
    assert parsed["action_name"] == "play_sequence"
    assert parsed["action_arguments"] == {"sequence_name": "basic_nod"}


def test_parse_markdown_json_code_block() -> None:
    raw = (
        "```json\n"
        '{\n  "observation": "看到猫",\n  "condition": "看到猫",\n'
        '  "action_name": "play_sequence",\n  "action_arguments": {"sequence_name": "basic_nod"}\n}\n'
        "```"
    )
    parsed = parse_conditional_fallback_json(raw)
    assert isinstance(parsed, dict)
    assert parsed["action_name"] == "play_sequence"

    # Also test generic ``` fence without language tag
    raw_generic = (
        "```\n"
        '{"observation": "看到猫", "condition": "看到猫", "action_name": "play_sequence", "action_arguments": {"sequence_name": "basic_nod"}}\n'
        "```"
    )
    parsed_generic = parse_conditional_fallback_json(raw_generic)
    assert parsed_generic["action_name"] == "play_sequence"


def test_parse_json_from_explanatory_text() -> None:
    chunks = [
        "你好，我已为你规划了条件任务：\n",
        '{"observation": "看到猫", "condition": "看到猫", ',
        '"action_name": "play_sequence", "action_arguments": {"sequence_name": "basic_nod"}}\n',
        "请确认执行！",
    ]
    parsed = parse_conditional_fallback_json(chunks)
    assert isinstance(parsed, dict)
    assert parsed["observation"] == "看到猫"
    assert parsed["action_name"] == "play_sequence"


def test_missing_json_raises_conditional_json_plan_missing() -> None:
    with pytest.raises(ValueError, match="conditional_json_plan_missing"):
        parse_conditional_fallback_json("这是一段纯文字回复，没有任何JSON对象。")

    with pytest.raises(ValueError, match="conditional_json_plan_missing"):
        parse_conditional_fallback_json("")

    with pytest.raises(json.JSONDecodeError):
        parse_conditional_fallback_json("前缀 { 坏掉的 json } 后缀")


def test_top_level_not_object_raises_conditional_json_plan_not_object() -> None:
    with pytest.raises(ValueError, match="conditional_json_plan_not_object"):
        parse_conditional_fallback_json("[1, 2, 3]")

    with pytest.raises(ValueError, match="conditional_json_plan_not_object"):
        parse_conditional_fallback_json("12345")

    with pytest.raises(ValueError, match="conditional_json_plan_not_object"):
        parse_conditional_fallback_json('"hello string"')

    with pytest.raises(ValueError, match="conditional_json_plan_not_object"):
        parse_conditional_fallback_json("```json\n[1, 2, 3]\n```")


def test_service_does_not_import_rclpy_or_have_side_effects() -> None:
    # Verify rclpy is not imported by the pure service
    assert "rclpy" not in sys.modules or "services.llm_conditional_planning" not in sys.modules.get("rclpy", {}).__name__
    import services.llm_conditional_planning as module
    source_code = open(module.__file__, "r", encoding="utf-8").read()
    assert "import rclpy" not in source_code
    assert "from rclpy" not in source_code
