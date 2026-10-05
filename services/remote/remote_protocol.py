"""Validation and ROS topic names for the browser remote-control protocol."""

from __future__ import annotations

import json
import math
from typing import Any


REMOTE_AUDIO_PCM_TOPIC = "/remote_audio_pcm"
REMOTE_VOICE_STATE_TOPIC = "/remote_voice_state"
REMOTE_SOURCE = "remote"
REMOTE_SAFETY_SOURCE = "remote_safety"
MAX_REMOTE_MESSAGE_BYTES = 64 * 1024
ALLOWED_ACTIONS = frozenset({
    "wave_hello",
    "raise_hand",
    "look_center",
    "happy_dance",
})


def _finite_unit(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and -1.0 <= numeric <= 1.0


def decode_remote_message(raw: str | bytes) -> dict[str, Any] | None:
    """Decode one bounded, allowlisted RTCDataChannel message."""
    if isinstance(raw, bytes):
        if len(raw) > MAX_REMOTE_MESSAGE_BYTES:
            return None
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_REMOTE_MESSAGE_BYTES:
        return None
    try:
        message = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(message, dict):
        return None
    message_type = message.get("type")
    seq = message.get("seq")
    if (
        not isinstance(message_type, str)
        or isinstance(seq, bool)
        or not isinstance(seq, int)
        or seq < 0
    ):
        return None

    if message_type == "control":
        vector = message.get("vector")
        if not isinstance(vector, dict) or set(vector) != {"forward", "turn", "yaw", "pitch"}:
            return None
        if not all(_finite_unit(vector.get(key)) for key in vector):
            return None
        return {
            "type": "control",
            "seq": seq,
            "vector": {key: float(vector[key]) for key in vector},
        }

    if message_type == "stop":
        reason = message.get("reason", "operator")
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            return None
        return {"type": "stop", "seq": seq, "reason": reason}

    if message_type == "action":
        name = message.get("name")
        if not isinstance(name, str) or name not in ALLOWED_ACTIONS:
            return None
        return {"type": "action", "seq": seq, "name": name}

    if message_type == "voice":
        state = message.get("state")
        if state not in {"start", "stop"}:
            return None
        return {"type": "voice", "seq": seq, "state": state}

    return None


def encode_voice_state(state: str) -> str:
    if state not in {"start", "stop"}:
        raise ValueError("voice state must be start or stop")
    return json.dumps({"state": state}, separators=(",", ":"), ensure_ascii=False)
