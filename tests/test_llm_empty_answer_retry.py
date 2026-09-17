"""Unit tests for pure LLM empty-answer retry request preparation and response parsing."""

from dataclasses import FrozenInstanceError
import inspect
from types import MappingProxyType
import pytest

from services.llm_empty_answer_retry import (
    EmptyAnswerRetryAccumulator,
    EmptyAnswerRetryRequest,
    build_empty_answer_retry_prompt,
    build_empty_answer_retry_request,
    calculate_retry_max_tokens,
    parse_retry_answer,
)


def test_retry_prompt_exact_verbatim():
    """1. Prompt must be strictly verbatim with Node implementation."""
    user_prompt = "今天北京天气如何？"
    prompt = build_empty_answer_retry_prompt(user_prompt)
    expected = (
        f"用户说：{user_prompt}\n"
        "请直接用一到两句简短自然的中文回答。只输出回答正文，不要输出纠错标签、"
        "分析过程、Markdown、动作说明或任何前缀。"
    )
    assert prompt == expected
    assert "用户说：今天北京天气如何？\n" in prompt
    assert prompt.endswith("分析过程、Markdown、动作说明或任何前缀。")


def test_retry_request_history_preserved():
    """2. history is preserved intact in the request."""
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮你的？"},
    ]
    req = build_empty_answer_retry_request("再见", history=history)
    assert req.history == history
    assert req.history is not history  # Defensive copy


def test_retry_request_tools_disabled():
    """3. tools_enabled must be False."""
    req = build_empty_answer_retry_request("测试")
    assert req.tools_enabled is False


def test_retry_request_structured_answer_disabled():
    """4. structured_answer must be False."""
    req = build_empty_answer_retry_request("测试")
    assert req.structured_answer is False


@pytest.mark.parametrize(
    ("input_value", "expected_tokens"),
    [
        # 缺少配置 -> 256
        (None, 256),
        ({}, 256),
        # 0 -> 256
        (0, 256),
        # 负数 -> 256
        (-1, 256),
        (-100, 256),
        # 1 -> 128
        (1, 128),
        # 127 -> 128
        (127, 128),
        # 128 -> 128
        (128, 128),
        # 200 -> 200
        (200, 200),
        # 256 -> 256
        (256, 256),
        # 512 -> 256
        (512, 256),
        # 字符串 -> 256
        ("200", 256),
        ("invalid", 256),
        # float -> 256
        (200.0, 256),
        (128.5, 256),
        # None -> 256
        (None, 256),
        # True -> 128 (isinstance(True, int) is True, max(1, 128) -> 128)
        (True, 128),
        # False -> 256 (isinstance(False, int) is True, 0 <= 0 -> 256)
        (False, 256),
    ],
)
def test_token_boundaries_direct_and_dict(input_value, expected_tokens):
    """5. Cover all token boundaries and Python bool inheritance semantics."""
    # Direct input
    assert calculate_retry_max_tokens(input_value) == expected_tokens

    # Dict input: {"max_tokens": input_value}
    if input_value is None:
        assert calculate_retry_max_tokens({"max_tokens": None}) == 256
        assert calculate_retry_max_tokens({}) == 256
    else:
        assert calculate_retry_max_tokens({"max_tokens": input_value}) == expected_tokens

    # Request builder integration
    settings = {"max_tokens": input_value} if input_value is not None else {}
    req = build_empty_answer_retry_request("测试", settings=settings)
    assert req.max_tokens_override == expected_tokens


def test_token_default_when_no_argument():
    """5b. Calling calculate_retry_max_tokens() without arguments returns 256."""
    assert calculate_retry_max_tokens() == 256


def test_token_settings_only_reads_real_dict():
    """The legacy node ignored non-dict settings objects."""
    settings = MappingProxyType({"max_tokens": 1})
    assert calculate_retry_max_tokens(settings) == 256


def test_accumulator_multiple_text_chunks_in_order():
    """6. Multiple text chunks are concatenated in order."""
    acc = EmptyAnswerRetryAccumulator()
    acc.process_event({"type": "text", "content": "今天天气"})
    acc.process_event({"type": "text", "content": "非常晴朗，"})
    acc.process_event({"type": "text", "content": "适合出门运动。"})
    assert acc.raw_text() == "今天天气非常晴朗，适合出门运动。"
    assert acc.parse_answer() == "今天天气非常晴朗，适合出门运动。"


def test_accumulator_ignores_non_text_events():
    """7. Non-text events (tool_call, dialog_expression, done, etc.) are ignored."""
    acc = EmptyAnswerRetryAccumulator()
    acc.process_event({"type": "tool_call", "name": "forward", "arguments": "{}"})
    acc.process_event({"type": "dialog_expression", "expression": "smile"})
    acc.process_event({"type": "done"})
    acc.process_event({"type": "other", "content": "should be ignored"})
    acc.process_event("invalid_format")
    acc.process_event(None)
    assert acc.raw_text() == ""
    assert acc.parse_answer() == ""


def test_accumulator_ignores_empty_content():
    """8. Events with empty or non-string content are ignored."""
    acc = EmptyAnswerRetryAccumulator()
    acc.process_event({"type": "text", "content": ""})
    acc.process_event({"type": "text", "content": None})
    acc.process_event({"type": "text", "content": 12345})
    acc.process_event({"type": "text"})
    assert acc.raw_text() == ""
    assert acc.parse_answer() == ""


def test_accumulator_empty_stream_returns_empty_string():
    """9. Empty event stream returns empty string."""
    acc = EmptyAnswerRetryAccumulator()
    assert acc.parse_answer() == ""
    assert parse_retry_answer("") == ""
    assert parse_retry_answer([]) == ""
    assert parse_retry_answer(None) == ""


def test_accumulator_whitespace_returns_empty_string():
    """10. Whitespace-only text returns empty string."""
    acc = EmptyAnswerRetryAccumulator()
    acc.process_event({"type": "text", "content": "   \n\t   "})
    assert acc.raw_text() == ""
    assert acc.parse_answer() == ""
    assert parse_retry_answer("   \n\t  ") == ""


def test_single_line_chinese_correction_returns_empty_string():
    """11. Single-line '【修正文本】: 你好' returns empty string to prevent reading correction label."""
    assert parse_retry_answer("【修正文本】: 你好") == ""
    assert parse_retry_answer("【纠错文本】: 你好") == ""


def test_single_line_bracket_correction_returns_empty_string():
    """12. Single-line '[corrected_text] 你好' returns empty string."""
    assert parse_retry_answer("[corrected_text] 你好") == ""
    assert parse_retry_answer("[corrected_text]: 你好") == ""


def test_multiline_correction_metadata_with_answer_returns_speech_only():
    """13. Multi-line correction metadata + answer returns speech text only."""
    text1 = "【修正文本】: 你好\n今天天气晴朗。"
    assert parse_retry_answer(text1) == "今天天气晴朗。"

    text2 = "【修正文本】\n你好\n今天天气晴朗。"
    assert parse_retry_answer(text2) == "今天天气晴朗。"

    text3 = "第一行: 修正文本: 你好\n第二行: 回答: 今天天气晴朗。"
    assert parse_retry_answer(text3) == "今天天气晴朗。"


def test_answer_prefix_stripped():
    """14. '回答：你好' is stripped to '你好'."""
    assert parse_retry_answer("回答：你好") == "你好"
    assert parse_retry_answer("回答: 你好") == "你好"
    assert parse_retry_answer("最终回答：你好呀") == "你好呀"


def test_markdown_and_special_symbols_cleaned():
    """15. Markdown formatting and unsafe symbols are sanitized via LLMResponsePolicy."""
    raw = "**今天**的天气*真好*！`rm -rf /`"
    cleaned = parse_retry_answer(raw)
    assert "*" not in cleaned
    assert "`" not in cleaned
    assert "/" not in cleaned
    assert "今天" in cleaned
    assert "的天气" in cleaned


def test_service_is_pure_and_does_not_import_rclpy():
    """16. Service is pure: does not import rclpy, call models, or publish messages."""
    import services.llm_empty_answer_retry as retry_mod

    assert "rclpy" not in retry_mod.__dict__
    source = inspect.getsource(retry_mod)
    assert "import rclpy" not in source
    assert "from rclpy" not in source
    assert "self.llm" not in source
    assert "chat_stream(" not in source
    assert "publish(" not in source


def test_frozen_request_cannot_be_reassigned():
    """17. Frozen request dataclass cannot be modified or reassigned."""
    req = build_empty_answer_retry_request("测试")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        req.prompt = "新提示词"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        req.tools_enabled = True  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        req.structured_answer = True  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        req.max_tokens_override = 128  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        req.history = []  # type: ignore[misc]
