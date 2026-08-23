"""Deterministic, non-speculative speech for accepted robot actions."""

from __future__ import annotations

import json


_SEQUENCE_ACKS = {
    "turn_head_left": "好的，我向左看。",
    "turn_head_right": "好的，我向右看。",
    "look_center": "好的，我看向前面。",
    "wave_hello": "好的，我向你招手。",
    "basic_wave": "好的，我向你招手。",
    "arms_up": "好的，我把手举起来。",
    "raise_hand": "好的，我举手。",
    "arms_down": "好的，我把手放下。",
    "basic_nod": "好的。",
    "happy_dance": "好呀。",
}

_MOVE_ACKS = {
    "forward": "好的，我向前走。",
    "backward": "好的，我向后退。",
    "spin": "好的，我转一圈。",
    "left": "好的，我向左转。",
    "right": "好的，我向右转。",
}

_TRACKING_ACKS = {
    "follow_me": "好的，我跟着你。",
    "look_at_me": "好的，我看着你。",
    "idle": "好的，已停止跟随。",
}


def _arguments(action):
    arguments = action.get("arguments", {}) if isinstance(action, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def action_acknowledgement(actions):
    """Return a short acknowledgement without claiming completion or perception."""
    if not isinstance(actions, (list, tuple)) or len(actions) != 1:
        return "好的。" if actions else ""
    action = actions[0]
    name = action.get("name") if isinstance(action, dict) else ""
    arguments = _arguments(action)
    if name == "play_sequence":
        return _SEQUENCE_ACKS.get(arguments.get("sequence_name"), "好的。")
    if name == "move_chassis":
        return _MOVE_ACKS.get(arguments.get("direction"), "好的。")
    if name == "set_tracking_mode":
        return _TRACKING_ACKS.get(arguments.get("mode"), "好的。")
    if name == "inspect_camera":
        return "好的，我看一下。"
    if name in {"express_emotion", "set_vision_gate"}:
        return "好呀。" if name == "express_emotion" else "好的。"
    return "好的。"
