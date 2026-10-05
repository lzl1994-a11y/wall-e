"""Small ROS protocol for interrupting foreground audio playback."""

from __future__ import annotations

import json


AUDIO_CONTROL_TOPIC = "/audio_playback_control"
STOP_SPEECH_ACTION = "stop_speech"


def encode_stop_speech_control(source: str = "remote_barge_in") -> str:
    return json.dumps(
        {"action": STOP_SPEECH_ACTION, "source": str(source)[:64]},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def decode_audio_control(raw: str) -> dict[str, str] | None:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("action") != STOP_SPEECH_ACTION:
        return None
    source = payload.get("source", "unknown")
    if not isinstance(source, str) or not source or len(source) > 64:
        return None
    if set(payload) - {"action", "source"}:
        return None
    return {"action": STOP_SPEECH_ACTION, "source": source}
