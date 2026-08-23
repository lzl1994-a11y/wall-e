"""统一工具调度器。

给 llm_service / voice_chat_service 提供：
  1. get_tools()              → OpenAI function calling 格式的工具列表
  2. ToolCallAccumulator      → 流式 tool_calls 碎片收集器
  3. build_action_cmd()       → 构造 /action_cmd 的 JSON 消息

来源统一为 mcp_service，不再各处重复定义。
"""

import json
import services.mcp_service as mcp
from services.action_command import build_action_cmd, parse_action_cmd


import yaml


DIRECT_ANSWER_TOOL_NAME = "direct_answer"
DIRECT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": DIRECT_ANSWER_TOOL_NAME,
        "description": (
            "每轮都必须调用一次。将唯一允许播放给用户听的最终台词写入 response。"
            "如需调用其他动作工具，也仍必须调用本工具。不要把台词写在普通 content 中。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "给用户播报的完整、自然、简短的最终台词",
                },
            },
            "required": ["response"],
            "additionalProperties": False,
        },
    },
}

MULTIMODAL_DIRECT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": DIRECT_ANSWER_TOOL_NAME,
        "description": (
            "每轮音频对话都必须调用一次。heard_text 写入你从本轮音频中听到的用户原话，"
            "response 写入唯一允许播放给用户听的最终台词。如需调用身体动作工具，也仍"
            "必须同时调用本工具。不要把台词写在普通 content 中。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "heard_text": {
                    "type": "string",
                    "description": "本轮音频中听到的用户原话，简洁转写，不要解释",
                    "maxLength": 240,
                },
                "response": {
                    "type": "string",
                    "description": "给用户播报的完整、自然、简短的最终台词",
                },
            },
            "required": ["heard_text", "response"],
            "additionalProperties": False,
        },
    },
}

def get_tools():
    """OpenAI function declarations, including the required speech outlet."""
    return [DIRECT_ANSWER_TOOL, *mcp.get_chat_tools()]


def get_multimodal_tools():
    """Tools for audio turns, including a transcript for paired history."""
    return [MULTIMODAL_DIRECT_ANSWER_TOOL, *mcp.get_chat_tools()]


class ToolCallAccumulator:
    """流式 tool_calls 碎片收集器。

    用法：
        acc = ToolCallAccumulator()
        for chunk in response:
            acc.feed(chunk.choices[0].delta)
        for tc in acc.flush():
            # tc = {"name": "...", "arguments": {...}}
    """

    def __init__(self):
        self._buffer = {}  # idx -> {"name": ..., "arguments": str}

    def feed(self, delta):
        if not delta.tool_calls:
            return
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in self._buffer:
                self._buffer[idx] = {"name": tc.function.name or "", "arguments": ""}
            else:
                if tc.function.name:
                    self._buffer[idx]["name"] = tc.function.name
            if tc.function.arguments:
                self._buffer[idx]["arguments"] += tc.function.arguments

    def flush(self):
        """返回已完成的 tool_call 列表，arguments 已解析为 dict。"""
        result = []
        for tc in sorted(self._buffer.values(), key=lambda x: x["name"]):
            raw = tc["arguments"].strip()
            if not raw:
                continue
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            result.append({"name": tc["name"], "arguments": args})
        return result
