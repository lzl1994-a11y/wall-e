"""Dependency-free ROS contracts for music playback and spectrum display."""

from __future__ import annotations

import json


MUSIC_AUDIO_TOPIC = "/music_audio"
MUSIC_SPECTRUM_TOPIC = "/music_spectrum"
MUSIC_STATE_TOPIC = "/music_state"


def encode_music_state(state: str, track: str = "", error: str = "") -> str:
    payload = {"state": str(state), "track": str(track)}
    if error:
        payload["error"] = str(error)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def decode_music_state(raw: str) -> dict[str, str] | None:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("state") not in {
        "loading", "playing", "stopped", "error",
    }:
        return None
    return {
        "state": payload["state"],
        "track": str(payload.get("track") or ""),
        "error": str(payload.get("error") or ""),
    }


__all__ = [
    "MUSIC_AUDIO_TOPIC", "MUSIC_SPECTRUM_TOPIC", "MUSIC_STATE_TOPIC",
    "decode_music_state", "encode_music_state",
]
