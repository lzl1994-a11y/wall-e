from services.llm.llm_stream_response import (
    StreamResponseAccumulator,
    StreamEventDecision,
    ACTION_FAILURE_REPLY,
    FALLBACK_EMPTY_REPLY,
)


def test_stream_response_accumulates_plain_text_sentences():
    accumulator = StreamResponseAccumulator()

    d1 = accumulator.process_event({"type": "text", "content": "你好！"})
    assert d1.tts_sentences == ["你好！"]

    d2 = accumulator.process_event({"type": "text", "content": "我是"})
    assert d2.tts_sentences == []

    d3 = accumulator.process_event({"type": "text", "content": "瓦力。"})
    assert d3.tts_sentences == ["我是瓦力。"]

    assert accumulator.text_buffer == "你好！我是瓦力。"
    assert accumulator.take_tail_tts() is None
    assert accumulator.decide_final_text() == "你好！我是瓦力。"


def test_first_long_clause_boundary_triggered_before_sentence_end():
    accumulator = StreamResponseAccumulator()

    # 长于 10 字符的首个分句（逗号）提前触发播报
    decision = accumulator.process_event({
        "type": "text",
        "content": "这是白居易琵琶行的开头两句，后面描写秋夜送客。",
    })
    assert decision.tts_sentences == ["这是白居易琵琶行的开头两句，", "后面描写秋夜送客。"]

    # 已经播报过之后，后续逗号不再提前切分，只在句末标点切分
    decision2 = accumulator.process_event({
        "type": "text",
        "content": "继续说，这是下一句。",
    })
    assert decision2.tts_sentences == ["继续说，这是下一句。"]


def test_short_comma_does_not_trigger_early_tts():
    accumulator = StreamResponseAccumulator()

    # "好的" 长度 < 10，不触发分句提前播报
    decision = accumulator.process_event({"type": "text", "content": "好的，我知道了。"})
    assert decision.tts_sentences == ["好的，我知道了。"]


def test_tail_tts_extracted_without_ending_punctuation():
    accumulator = StreamResponseAccumulator()

    d1 = accumulator.process_event({"type": "text", "content": "你好，这是没有结尾标点的内容"})
    assert d1.tts_sentences == []

    tail = accumulator.take_tail_tts()
    assert tail == "你好，这是没有结尾标点的内容"
    assert accumulator.take_tail_tts() is None


def test_dialog_expression_event():
    accumulator = StreamResponseAccumulator()

    decision = accumulator.process_event({
        "type": "dialog_expression",
        "expression": "happy",
        "intensity": "high",
    })
    assert decision.expression == {"expression": "happy", "intensity": "high"}


def test_tool_call_and_done_events():
    accumulator = StreamResponseAccumulator()

    decision = accumulator.process_event({
        "type": "tool_call",
        "name": "play_sequence",
        "arguments": '{"sequence_name": "wave_hello"}',
    })
    assert decision.tool_call == {
        "name": "play_sequence",
        "arguments": '{"sequence_name": "wave_hello"}',
    }

    decision_done = accumulator.process_event({
        "type": "done",
        "finish_reason": "stop",
    })
    assert decision_done.finish_reason == "stop"
    assert accumulator.finish_reason == "stop"


def test_final_text_action_acknowledgement():
    accumulator = StreamResponseAccumulator()

    # 纯动作场景：模型无说话文本，执行动作后按动作生成回执确认语
    final_text = accumulator.decide_final_text(
        actions=[{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}]
    )
    assert final_text == "好的，我向你招手。"


def test_final_text_action_failure():
    accumulator = StreamResponseAccumulator()

    final_text = accumulator.decide_final_text(action_failure=True)
    assert final_text == ACTION_FAILURE_REPLY


def test_final_text_rejected_actions():
    accumulator = StreamResponseAccumulator()

    accumulator.record_rejected_action("turn_head", "invalid_arguments")
    assert accumulator.decide_final_text() == "我不太确定你是不是要我执行这个动作，可以再明确说一下吗？"

    # 包含摄像头检查被拒
    accumulator.record_rejected_action("inspect_camera", "disallowed")
    assert accumulator.decide_final_text() == "你是想让我打开摄像头看一下吗？"


def test_single_line_correction_tag_produces_empty_final_text():
    accumulator = StreamResponseAccumulator()

    accumulator.process_event({"type": "text", "content": "【修正文本】背一下琵琶行"})
    assert accumulator.decide_final_text() == ""


def test_streaming_tts_keeps_legacy_character_only_cleaning():
    accumulator = StreamResponseAccumulator()

    decision = accumulator.process_event({
        "type": "text", "content": "[corrected_text] 你好！"
    })

    assert decision.tts_sentences == ["corrected_text 你好！"]
