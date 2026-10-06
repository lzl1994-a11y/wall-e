"""Validation and ROS topic names for the browser remote-control protocol."""

from __future__ import annotations

import json
import math
from typing import Any


# Legacy remote-ASR topics are kept for the local voice pipelines. The WebRTC
# intercom uses dedicated topics so live audio cannot accidentally be fed into
# ASR/LLM or echoed back through the browser track.
REMOTE_AUDIO_PCM_TOPIC = "/remote_audio_pcm"
REMOTE_VOICE_STATE_TOPIC = "/remote_voice_state"
ROBOT_AUDIO_PCM_TOPIC = "/robot_audio_pcm"
REMOTE_AUDIO_PLAYBACK_TOPIC = "/remote_audio_playback"
REMOTE_INTERCOM_STATE_TOPIC = "/remote_intercom_state"
REMOTE_SOURCE = "remote"
REMOTE_SAFETY_SOURCE = "remote_safety"
MAX_REMOTE_MESSAGE_BYTES = 64 * 1024
MAX_REMOTE_SEQUENCE = 2**63 - 1
ALLOWED_ACTIONS = frozenset({
    "wave_hello",
    "raise_hand",
    "look_center",
    "happy_dance",
})


def action_request_for(name: str) -> tuple[str, dict[str, str]]:
    """Map a browser alias to the existing canonical action skill."""
    if name not in ALLOWED_ACTIONS:
        raise ValueError("action is not allowed")
    return "play_sequence", {"sequence_name": name}


def _finite_unit(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and -1.0 <= numeric <= 1.0


def _reject_json_constant(value: str) -> None:
    """Reject JSON extensions that Python's decoder accepts by default."""
    raise ValueError(f"non-finite JSON constant: {value}")


def decode_remote_message(raw: str | bytes) -> dict[str, Any] | None:
    """Decode one bounded, allowlisted RTCDataChannel message."""
    if isinstance(raw, bytes):
        if len(raw) > MAX_REMOTE_MESSAGE_BYTES:
            return None
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str):
        return None
    try:
        if len(raw.encode("utf-8")) > MAX_REMOTE_MESSAGE_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    try:
        message = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError):
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
        or seq > MAX_REMOTE_SEQUENCE
    ):
        return None

    if message_type == "control":
        if set(message) != {"type", "seq", "vector"}:
            return None
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
        if set(message) not in ({"type", "seq"}, {"type", "seq", "reason"}):
            return None
        reason = message.get("reason", "operator")
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            return None
        return {"type": "stop", "seq": seq, "reason": reason}

    if message_type == "action":
        if set(message) != {"type", "seq", "name"}:
            return None
        name = message.get("name")
        if not isinstance(name, str) or name not in ALLOWED_ACTIONS:
            return None
        return {"type": "action", "seq": seq, "name": name}

    if message_type == "call":
        if set(message) != {"type", "seq", "state"}:
            return None
        state = message.get("state")
        if state not in {"start", "end"}:
            return None
        return {"type": "call", "seq": seq, "state": state}

    if message_type == "voice":
        if set(message) != {"type", "seq", "state"}:
            return None
        state = message.get("state")
        if state not in {"start", "stop"}:
            return None
        return {"type": "voice", "seq": seq, "state": state}

    return None


def encode_voice_state(state: str) -> str:
    if state not in {"start", "stop"}:
        raise ValueError("voice state must be start or stop")
    return json.dumps({"state": state}, separators=(",", ":"), ensure_ascii=False)


def encode_call_state(state: str) -> str:
    if state not in {"start", "end"}:
        raise ValueError("call state must be start or end")
    return json.dumps(
        {"state": state, "mode": "session"},
        separators=(",", ":"),
        ensure_ascii=False,
    )
