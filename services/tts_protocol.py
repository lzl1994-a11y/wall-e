"""Control messages carried on the ordered ``tts_text`` ROS topic."""

from __future__ import annotations

import json


_CONTROL_FIELD = "_wali_tts_control"
_TURN_END = "turn_end"


def encode_turn_end(turn_id: str) -> str:
    """Return an ordered marker indicating that a TTS turn is complete."""
    return json.dumps(
        {_CONTROL_FIELD: _TURN_END, "turn_id": str(turn_id or "")},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def decode_turn_end(message: str) -> str | None:
    """Return the turn ID for a valid end marker, otherwise ``None``."""
    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get(_CONTROL_FIELD) != _TURN_END:
        return None
    turn_id = payload.get("turn_id")
    return turn_id if isinstance(turn_id, str) else None
