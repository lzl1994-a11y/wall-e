import json
from collections import deque

from services.llm.llm_conversation_history import (
    LLMConversationHistory,
    DEFAULT_CHAT_HISTORY_MESSAGES,
    DEFAULT_VISUAL_HISTORY_MESSAGES,
)


def test_build_request_history_limits_count_and_starts_with_user():
    raw_history = [{"role": "assistant", "content": "orphan_start"}]
    for i in range(10):
        raw_history.append({"role": "user", "content": f"u{i}"})
        raw_history.append({"role": "assistant", "content": f"a{i}"})

    selected = LLMConversationHistory.build_request_history(
        raw_history, max_messages=DEFAULT_CHAT_HISTORY_MESSAGES
    )

    assert len(selected) <= DEFAULT_CHAT_HISTORY_MESSAGES
    assert selected[0]["role"] == "user"
    assert selected[-1]["content"] == "a9"


def test_build_request_history_drops_image_blocks_permanently():
    raw_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "上一轮看到了什么"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,test-image-data"},
                },
            ],
        },
        {"role": "assistant", "content": "前面有一本书。"},
    ]

    selected = LLMConversationHistory.build_request_history(raw_history)
    assert selected[0]["content"] == "上一轮看到了什么"
    assert "test-image-data" not in repr(selected)
    assert selected[1]["content"] == "前面有一本书。"


def test_build_visual_history_only_keeps_text_user_and_assistant():
    raw_history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": '{"status": "completed"}'},
        {"role": "assistant", "content": "好的，动作完成。"},
        {"role": "user", "content": "  "},  # 空白文本应丢弃
        {"role": "system", "content": "系统指令"},  # 其他角色丢弃
        {"role": "user", "content": "现在能看到什么"},
    ]

    selected = LLMConversationHistory.build_visual_history(
        raw_history, max_messages=DEFAULT_VISUAL_HISTORY_MESSAGES
    )

    assert [item["role"] for item in selected] == ["user", "assistant", "user"]
    assert selected[0]["content"] == "你好"
    assert selected[1]["content"] == "好的，动作完成。"
    assert selected[2]["content"] == "现在能看到什么"


def test_record_dialog_turn():
    history = deque(maxlen=20)
    LLMConversationHistory.record_dialog_turn(
        history,
        user_text="今天天气怎么样",
        assistant_text="今天天气很晴朗。",
    )

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "今天天气怎么样"}
    assert history[1] == {"role": "assistant", "content": "今天天气很晴朗。"}


def test_record_action_turn_protocol_order_and_tool_call_ids():
    history = deque(maxlen=20)
    actions = [
        {
            "name": "play_sequence",
            "arguments": '{"sequence_name": "wave_hello"}',
            "status": "completed",
            "reason": "",
        }
    ]

    LLMConversationHistory.record_action_turn(
        history,
        user_text="挥挥手",
        assistant_text="好的，我向你招手。",
        actions=actions,
        turn_id="turn-123",
    )

    items = list(history)
    assert [m["role"] for m in items] == ["user", "assistant", "tool", "assistant"]
    assert items[0] == {"role": "user", "content": "挥挥手"}

    # assistant tool_calls
    assert items[1]["content"] is None
    assert len(items[1]["tool_calls"]) == 1
    assert items[1]["tool_calls"][0]["id"] == "call_turn-123_0"
    assert items[1]["tool_calls"][0]["function"]["name"] == "play_sequence"

    # tool response
    assert items[2]["role"] == "tool"
    assert items[2]["tool_call_id"] == "call_turn-123_0"
    assert items[2]["name"] == "play_sequence"
    payload = json.loads(items[2]["content"])
    assert payload["status"] == "completed"

    # final assistant reply
    assert items[3] == {"role": "assistant", "content": "好的，我向你招手。"}


def test_record_turn_delegation():
    h1 = deque()
    LLMConversationHistory.record_turn(
        h1, user_text="问句", assistant_text="答句", actions=None
    )
    assert len(h1) == 2

    h2 = deque()
    LLMConversationHistory.record_turn(
        h2,
        user_text="动作指令",
        assistant_text="好的。",
        actions=[{"name": "look_center", "status": "completed"}],
        turn_id="t99",
    )
    assert len(h2) == 4
    assert [m["role"] for m in h2] == ["user", "assistant", "tool", "assistant"]
