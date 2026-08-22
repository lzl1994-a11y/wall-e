"""Small, dependency-free `/action_cmd` payload codec.

This module intentionally uses only the standard library so the motion/handheld
control path can start even when optional LLM, FastMCP, or cloud dependencies
are unavailable.
"""

from __future__ import annotations

import json
from typing import Any


def build_action_cmd(tool_name: str, arguments: Any) -> str:
    """Build a JSON action command from dict or JSON-string arguments."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(tool_name, str) or not tool_name or not isinstance(arguments, dict):
        raise ValueError("action_cmd requires a tool name and object arguments")
    return json.dumps({"name": tool_name, "arguments": arguments})


def parse_action_cmd(payload: Any) -> tuple[str, dict[str, Any]] | None:
    """Decode an action command; return ``None`` for malformed messages."""
    try:
        command = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(command, dict) or not isinstance(command.get("name"), str):
        return None
    arguments = command.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return command["name"], arguments
