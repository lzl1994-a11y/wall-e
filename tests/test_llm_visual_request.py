"""Unit tests for services.llm.llm_visual_request."""

from __future__ import annotations

import base64
import dataclasses
import sys
import pytest

from services.llm.llm_visual_request import (
    GAME_VISION_MAX_TOKENS,
    GAME_VISION_PROMPT,
    GAME_VISION_SYSTEM_PROMPT,
    CAMERA_QA_SYSTEM_PROMPT,
    CONDITIONAL_VISION_MAX_TOKENS,
    CONDITIONAL_VISION_SYSTEM_PROMPT,
    LLMVisualRequest,
    VisualModelRequest,
    VisualResponseAccumulator,
    build_camera_qa_prompt,
    build_camera_qa_request,
    build_conditional_vision_prompt,
    build_conditional_vision_request,
    build_game_vision_request,
    encode_image_base64,
)


def test_jpeg_bytes_encoding_matches_base64_ascii() -> None:
    sample_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
    encoded = encode_image_base64(sample_bytes)
    expected = base64.b64encode(sample_bytes).decode("ascii")
    assert encoded == expected
    assert isinstance(encoded, str)

    # bytearray is also supported
    assert encode_image_base64(bytearray(sample_bytes)) == expected

    with pytest.raises(TypeError):
        build_game_vision_request(expected)  # type: ignore[arg-type]


def test_game_request_prompt_verbatim_matches() -> None:
    sample_jpeg = b"\xff\xd8\xff\xe0fake_jpeg"
    req = build_game_vision_request(sample_jpeg)
    expected_prompt = (
        "观察这张正在运行的 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
        "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。只输出可直接播报的一句话。"
    )
    expected_system_prompt = (
        "你是陪主人玩 FC 游戏的瓦力。只依据当前游戏截图简短评论，不输出分析过程。"
    )
    assert req.prompt == expected_prompt
    assert req.system_prompt == expected_system_prompt


def test_game_request_parameters() -> None:
    sample_jpeg = b"\xff\xd8\xff\xe0fake_jpeg"
    req = build_game_vision_request(sample_jpeg)
    assert req.history == []
    assert req.tools_enabled is False
    assert req.structured_answer is False
    assert req.max_tokens_override == 96
    assert req.image_base64 == base64.b64encode(sample_jpeg).decode("ascii")


def test_camera_qa_prompt_includes_user_prompt_verbatim() -> None:
    user_prompt = "帮我看看桌上有几个苹果？"
    prompt = build_camera_qa_prompt(user_prompt)
    expected_prompt = (
        "请根据我附带的摄像头画面回答用户的问题。只输出简短、自然、可直接播报的中文答案，"
        "不要输出修正文本标签、分析过程、工具调用或括号说明。\n"
        f"用户问题：{user_prompt}"
    )
    assert prompt == expected_prompt

    sample_frame = b"fake_frame_bytes"
    req = build_camera_qa_request(sample_frame, user_prompt=user_prompt)
    assert req.prompt == expected_prompt
    assert req.system_prompt == (
        "你是瓦力的视觉。只依据当前摄像头图片回答问题；看不清时明确说看不清，"
        "不要猜测。答案使用简短自然的中文，不能输出分析过程或任何标签。"
    )


def test_camera_qa_retains_provided_visual_history() -> None:
    history = [
        {"role": "user", "content": "之前看到了什么？"},
        {"role": "assistant", "content": "之前看到了一只小猫。"},
    ]
    sample_frame = b"fake_frame_bytes"
    req = build_camera_qa_request(
        sample_frame,
        user_prompt="现在小猫还在吗？",
        history=history,
    )
    assert req.history == history
    # Modifying original passed history does not alter req if constructed with list copy
    assert req.history is not history or req.history == history


def test_camera_qa_max_tokens_override_is_none() -> None:
    req = build_camera_qa_request(b"frame", user_prompt="你好")
    assert req.max_tokens_override is None
    assert req.tools_enabled is False
    assert req.structured_answer is False


def test_conditional_vision_prompt_inserts_observation_and_condition() -> None:
    observation = "观察前方地面"
    condition = "地面上有障碍物"
    prompt = build_conditional_vision_prompt(observation, condition)
    expected_prompt = (
        "请只依据附带的当前摄像头画面判断条件。返回一个 JSON 对象，且只能包含 "
        "decision 和 evidence。decision 只能是 yes、no、uncertain；图片不足以确认时"
        "必须使用 uncertain。不要执行动作，不要输出 Markdown 或其他文字。\n"
        f"观察任务：{observation}\n判断条件：{condition}"
    )
    assert prompt == expected_prompt

    req = build_conditional_vision_request(
        b"frame", observation=observation, condition=condition
    )
    assert req.prompt == expected_prompt
    assert req.system_prompt == (
        "你是机器人视觉条件判断器。只能依据当前图片返回严格 JSON；"
        "无法确认时必须返回 uncertain，禁止猜测。"
    )


def test_conditional_vision_parameters() -> None:
    req = build_conditional_vision_request(
        b"frame_bytes",
        observation="观察前方",
        condition="有人挥手",
    )
    assert req.history == []
    assert req.tools_enabled is False
    assert req.structured_answer is False
    assert req.max_tokens_override == 160
    assert req.image_base64 == base64.b64encode(b"frame_bytes").decode("ascii")


def test_accumulator_only_accepts_valid_text_events() -> None:
    accumulator = VisualResponseAccumulator()
    accumulator.process_event({"type": "tool_call", "name": "test"})
    accumulator.process_event({"type": "text", "content": ""})
    accumulator.process_event({"type": "text", "content": None})
    accumulator.process_event({"type": "done"})
    accumulator.process_event("invalid_event_type")
    accumulator.process_event(None)
    assert accumulator.raw_text() == ""
    assert accumulator.clean_answer() == ""

    accumulator.process_event({"type": "text", "content": "瓦力看到"})
    accumulator.process_event({"type": "text", "content": "了一本书。"})
    assert accumulator.raw_text() == "瓦力看到了一本书。"


def test_accumulator_raw_text_concatenates_and_strips() -> None:
    accumulator = VisualResponseAccumulator()
    accumulator.process_event({"type": "text", "content": "  {\n"})
    accumulator.process_event(
        {"type": "text", "content": '  "decision": "yes",\n  "evidence": "cat seen"\n'}
    )
    accumulator.process_event({"type": "text", "content": "}   \n"})
    raw = accumulator.raw_text()
    assert raw.startswith("{")
    assert raw.endswith("}")
    assert '"decision": "yes"' in raw


def test_accumulator_clean_answer_reuses_response_policy() -> None:
    accumulator = VisualResponseAccumulator()
    # Markdown code block with visual prefix
    accumulator.process_event(
        {"type": "text", "content": "```\n【修正文本】 我看到前面有一把椅子。\n```"}
    )
    cleaned = accumulator.clean_answer()
    assert cleaned == "我看到前面有一把椅子。"


def test_empty_event_stream_returns_empty_strings() -> None:
    accumulator = VisualResponseAccumulator()
    assert accumulator.raw_text() == ""
    assert accumulator.clean_answer() == ""


def test_service_does_not_import_rclpy_or_have_side_effects() -> None:
    import services.llm.llm_visual_request as module

    source = open(module.__file__, "r", encoding="utf-8").read()
    assert "import rclpy" not in source
    assert "from rclpy" not in source
    assert "rclpy" not in sys.modules or "services.llm.llm_visual_request" not in sys.modules.get("rclpy", {}).__name__


def test_frozen_request_object_is_immutable() -> None:
    req = build_game_vision_request(b"test")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.prompt = "modified_prompt"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.max_tokens_override = 128  # type: ignore[misc]
