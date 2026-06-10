"""统一工具调度器。

给 llm_service / voice_chat_service 提供：
  1. get_tools()              → OpenAI function calling 格式的工具列表
  2. ToolCallAccumulator      → 流式 tool_calls 碎片收集器
  3. build_action_cmd()       → 构造 /action_cmd 的 JSON 消息

来源统一为 mcp_service，不再各处重复定义。
"""

import json
import services.mcp_service as mcp


def get_tools():
    """OpenAI function calling 格式的工具列表"""
    return mcp.get_chat_tools()


def build_action_cmd(tool_name, arguments):
    """将 LLM 返回的 tool_call 构造为 /action_cmd 的 payload。

    arguments 可以是 dict 或 JSON 字符串。
    """
    if isinstance(arguments, str):
        args = json.loads(arguments or "{}")
    else:
        args = arguments
    return json.dumps({"name": tool_name, "arguments": args})


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