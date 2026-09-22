"""Unit tests for services.llm.llm_voice_turn."""

from __future__ import annotations

import json
import sys
import pytest

from services.llm.llm_voice_turn import (
    FALLBACK_EMPTY_REPLY,
    LLMVoiceTurnState,
    VoiceTurnFinalDecision,
)


def test_normalize_corrected_text_normal() -> None:
    state = LLMVoiceTurnState("你好瓦力")
    assert state.corrected_text == ""
    assert state.corrected_text_published is False

    normalized = state.normalize_corrected_text("  你好，瓦力机器人  ")
    assert normalized == "你好，瓦力机器人"
    assert state.corrected_text == "你好，瓦力机器人"
    assert state.corrected_text_published is True


def test_normalize_corrected_text_fallback_to_user_prompt() -> None:
    state = LLMVoiceTurnState("前进两米")
    assert state.normalize_corrected_text("") == "前进两米"
    assert state.corrected_text == "前进两米"

    state2 = LLMVoiceTurnState("后退一步")
    assert state2.normalize_corrected_text(None) == "后退一步"

    state3 = LLMVoiceTurnState("左转")
    assert state3.normalize_corrected_text("    ") == "左转"


def test_corrected_text_published_state() -> None:
    state = LLMVoiceTurnState("测试")
    assert state.corrected_text_published is False
    state.normalize_corrected_text("测试修正")
    assert state.corrected_text_published is True


def test_default_expression_initial_needed() -> None:
    state = LLMVoiceTurnState("测试")
    assert state.needs_default_expression is True
    assert state.expression_published is False


def test_mark_expression_published_disables_default_expression() -> None:
    state = LLMVoiceTurnState("测试")
    assert state.needs_default_expression is True
    state.mark_expression_published()
    assert state.needs_default_expression is False
    assert state.expression_published is True


def test_process_event_delegation() -> None:
    state = LLMVoiceTurnState("你好")
    decision = state.process_event({"type": "text", "content": "收到，我在。"})
    assert decision.tts_sentences == ["收到，我在。"]


def test_record_spoken_saves_spoken_parts() -> None:
    state = LLMVoiceTurnState("你好")
    assert state.accumulator.spoken_parts == []
    state.record_spoken("收到，我在。")
    assert state.accumulator.spoken_parts == ["收到，我在。"]


def test_record_rejected_action_affects_final_decision() -> None:
    state = LLMVoiceTurnState("走十米")
    state.record_rejected_action("move_chassis", "safety_limit")
    decision = state.decide_final_turn()
    assert decision.clean_text == "我不太确定你是不是要我执行这个动作，可以再明确说一下吗？"

    state_cam = LLMVoiceTurnState("看一眼")
    state_cam.record_rejected_action("inspect_camera", "argument_conflict")
    decision_cam = state_cam.decide_final_turn()
    assert decision_cam.clean_text == "你是想让我打开摄像头看一下吗？"


def test_record_pending_action_payload_format() -> None:
    state = LLMVoiceTurnState("招手")
    turn_id = "turn-123"
    arguments = {"sequence_name": "wave_hello"}
    payload = state.record_pending_action(turn_id, "play_sequence", arguments)

    assert payload == {
        "turn_id": "turn-123",
        "name": "play_sequence",
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }
    assert state.actions == [payload]


def test_pending_actions_preserves_raw_dict() -> None:
    state = LLMVoiceTurnState("招手")
    args = {"sequence_name": "wave_hello"}
    state.record_pending_action("turn-1", "play_sequence", args)

    assert state.pending_actions == [{
        "name": "play_sequence",
        "arguments": args,
    }]
    assert isinstance(state.pending_actions[0]["arguments"], dict)


def test_action_payload_ensure_ascii_false() -> None:
    state = LLMVoiceTurnState("播报测试")
    args = {"text": "你好世界，中文测试"}
    payload = state.record_pending_action("turn-1", "custom_action", args)
    assert "你好世界，中文测试" in payload["arguments"]
    assert "\\u" not in payload["arguments"]


def test_apply_action_sequence_normalization() -> None:
    state = LLMVoiceTurnState("测试")
    raw_sequence = {
        "results": [
            {
                "name": "wave_hello",
                "arguments": {"speed": 1},
                "status": "completed",
                "request_id": "req-1",
                "reason": "",
            }
        ]
    }
    actions = state.apply_action_sequence(raw_sequence, "turn-1")
    assert len(actions) == 1
    assert actions[0]["turn_id"] == "turn-1"
    assert actions[0]["name"] == "wave_hello"
    assert actions[0]["status"] == "completed"
    assert actions[0]["request_id"] == "req-1"
    assert actions[0]["arguments"] == json.dumps({"speed": 1}, ensure_ascii=False)
    assert state.actions == actions
    assert state.action_failure is None


def test_apply_action_sequence_fallback_action_to_name() -> None:
    state = LLMVoiceTurnState("测试")
    raw_sequence = {
        "results": [
            {
                "action": "wave_fallback",
                "status": "completed",
            }
        ]
    }
    actions = state.apply_action_sequence(raw_sequence, "turn-1")
    assert actions[0]["name"] == "wave_fallback"


def test_apply_action_sequence_default_status_failed() -> None:
    state = LLMVoiceTurnState("测试")
    raw_sequence = {
        "results": [
            {
                "name": "test_action",
            }
        ]
    }
    actions = state.apply_action_sequence(raw_sequence, "turn-1")
    assert actions[0]["status"] == "failed"
    assert state.action_failure is not None
    assert state.action_failure.get("name") == "test_action"


def test_apply_action_sequence_first_failure_detection() -> None:
    state = LLMVoiceTurnState("测试")
    raw_sequence = {
        "results": [
            {"name": "step1", "status": "completed"},
            {"name": "step2", "status": "failed", "reason": "timeout"},
            {"name": "step3", "status": "failed", "reason": "cancelled"},
        ]
    }
    state.apply_action_sequence(raw_sequence, "turn-1")
    assert state.action_failure is not None
    assert state.action_failure["name"] == "step2"
    assert state.action_failure["reason"] == "timeout"


def test_apply_action_sequence_all_completed_skipped_no_failure() -> None:
    state = LLMVoiceTurnState("测试")
    raw_sequence = {
        "results": [
            {"name": "step1", "status": "completed"},
            {"name": "step2", "status": "skipped"},
        ]
    }
    state.apply_action_sequence(raw_sequence, "turn-1")
    assert state.action_failure is None


def test_final_user_text_prefers_corrected_text() -> None:
    state = LLMVoiceTurnState("原始ASR文本")
    state.normalize_corrected_text("修正后的文本")
    decision = state.decide_final_turn()
    assert decision.user_text == "修正后的文本"

    state_uncorrected = LLMVoiceTurnState("未修正的文本")
    decision_uncorrected = state_uncorrected.decide_final_turn()
    assert decision_uncorrected.user_text == "未修正的文本"


def test_empty_final_text_requires_retry() -> None:
    state = LLMVoiceTurnState("你好")
    decision = state.decide_final_turn()
    assert decision.clean_text == ""
    assert decision.needs_empty_answer_retry is True


def test_resolve_final_text_uses_fallback_when_retry_empty() -> None:
    state = LLMVoiceTurnState("你好")
    resolved_empty = state.resolve_final_text("")
    assert resolved_empty == FALLBACK_EMPTY_REPLY

    resolved_none = state.resolve_final_text(None)
    assert resolved_none == FALLBACK_EMPTY_REPLY

    resolved_whitespace = state.resolve_final_text("   ")
    assert resolved_whitespace == FALLBACK_EMPTY_REPLY

    resolved_valid = state.resolve_final_text("你好，我是瓦力！")
    assert resolved_valid == "你好，我是瓦力！"


def test_spoken_parts_prevents_final_tts() -> None:
    state = LLMVoiceTurnState("你好")
    state.record_spoken("你好，主人。")
    decision = state.decide_final_turn()
    assert decision.should_publish_final_tts is False


def test_tail_after_record_spoken_prevents_duplicate_final_tts() -> None:
    state = LLMVoiceTurnState("你好")
    state.process_event({"type": "text", "content": "好的，马上办"})
    tail = state.take_tail_tts()
    assert tail == "好的，马上办"

    # Node speaks tail and records it
    state.record_spoken(tail)

    decision = state.decide_final_turn()
    assert decision.clean_text == "好的，马上办"
    assert decision.should_publish_final_tts is False


def test_service_does_not_import_rclpy_or_have_side_effects() -> None:
    import services.llm.llm_voice_turn as module

    source = open(module.__file__, "r", encoding="utf-8").read()
    assert "import rclpy" not in source
    assert "from rclpy" not in source
    assert "rclpy" not in sys.modules or "services.llm.llm_voice_turn" not in sys.modules.get("rclpy", {}).__name__
