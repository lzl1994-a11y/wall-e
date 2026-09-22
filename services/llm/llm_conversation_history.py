"""Pure service for filtering, cleaning, and recording LLM conversation history."""

from __future__ import annotations

import json
from typing import Any, MutableSequence, Sequence


DEFAULT_CHAT_HISTORY_MESSAGES = 12
DEFAULT_VISUAL_HISTORY_MESSAGES = 8


class LLMConversationHistory:
    """Pure conversation history transformation and recording policies."""

    @staticmethod
    def text_only_history_message(item: Any) -> dict[str, Any] | None:
        """Copy one history item while permanently dropping image blocks."""
        if not isinstance(item, dict):
            return None
        message = dict(item)
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                value = block.get("text")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            message["content"] = "\n".join(text_parts)
        elif content is not None and not isinstance(content, str):
            message["content"] = ""
        return message

    @classmethod
    def build_request_history(
        cls,
        history: Sequence[dict[str, Any]],
        *,
        max_messages: int = DEFAULT_CHAT_HISTORY_MESSAGES,
    ) -> list[dict[str, Any]]:
        """Construct request history bounded to recent messages, starting with user."""
        selected = [
            message
            for item in list(history)[-max_messages:]
            if (message := cls.text_only_history_message(item)) is not None
        ]
        while selected and selected[0].get("role") != "user":
            selected.pop(0)
        return selected

    @staticmethod
    def build_visual_history(
        history: Sequence[dict[str, Any]],
        *,
        max_messages: int = DEFAULT_VISUAL_HISTORY_MESSAGES,
    ) -> list[dict[str, Any]]:
        """Construct visual request history with only text user/assistant messages."""
        selected: list[dict[str, Any]] = []
        for item in list(history)[-max_messages:]:
            if not isinstance(item, dict):
                continue
            if item.get("role") not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                selected.append({"role": item["role"], "content": content})
        return selected

    @staticmethod
    def record_dialog_turn(
        history: MutableSequence[dict[str, Any]],
        *,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Append a standard user and assistant dialogue turn to history."""
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})

    @staticmethod
    def record_action_turn(
        history: MutableSequence[dict[str, Any]],
        *,
        user_text: str,
        assistant_text: str,
        actions: Sequence[dict[str, Any]],
        turn_id: str = "",
    ) -> None:
        """Append an OpenAI-compliant tool interaction turn to history."""
        history.append({"role": "user", "content": user_text})

        openai_tool_calls: list[dict[str, Any]] = []
        for index, act in enumerate(actions):
            call_id = f"call_{turn_id}_{index}"
            if isinstance(act, dict):
                act["id"] = call_id
                action_name = act.get("name", "")
                arguments = act.get("arguments", "{}")
            else:
                action_name = getattr(act, "name", "")
                arguments = getattr(act, "arguments", "{}")
            openai_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": action_name,
                    "arguments": arguments,
                },
            })

        history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": openai_tool_calls,
        })

        for act in actions:
            if isinstance(act, dict):
                tool_call_id = act.get("id", "")
                action_name = act.get("name", "")
                status = act.get("status", "accepted")
                reason = act.get("reason", "")
            else:
                tool_call_id = getattr(act, "id", "")
                action_name = getattr(act, "name", "")
                status = getattr(act, "status", "accepted")
                reason = getattr(act, "reason", "")

            history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": action_name,
                "content": json.dumps({
                    "status": status,
                    "reason": reason,
                }, ensure_ascii=False),
            })

        history.append({"role": "assistant", "content": assistant_text})

    @classmethod
    def record_turn(
        cls,
        history: MutableSequence[dict[str, Any]],
        *,
        user_text: str,
        assistant_text: str,
        actions: Sequence[dict[str, Any]] | None = None,
        turn_id: str = "",
    ) -> None:
        """Record either a plain dialog turn or an action tool turn."""
        if actions:
            cls.record_action_turn(
                history,
                user_text=user_text,
                assistant_text=assistant_text,
                actions=actions,
                turn_id=turn_id,
            )
        else:
            cls.record_dialog_turn(
                history,
                user_text=user_text,
                assistant_text=assistant_text,
            )


ConversationHistoryService = LLMConversationHistory
LLMConversationHistoryService = LLMConversationHistory

__all__ = [
    "ConversationHistoryService",
    "DEFAULT_CHAT_HISTORY_MESSAGES",
    "DEFAULT_VISUAL_HISTORY_MESSAGES",
    "LLMConversationHistory",
    "LLMConversationHistoryService",
]
