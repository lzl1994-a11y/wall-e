"""Correlated PCM requests for short, non-dialogue system prompts."""

from __future__ import annotations

import base64
import binascii
import json


SYSTEM_AUDIO_TOPIC = "system_audio_output"
SYSTEM_AUDIO_DONE_TOPIC = "system_audio_done"


def encode_system_audio(request_id: str, cue: str, pcm: bytes) -> str:
    """Encode one complete 48 kHz, mono, signed-16-bit system prompt."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("System audio requires a request ID")
    if not isinstance(cue, str) or not cue.strip():
        raise ValueError("System audio requires a cue name")
    if not pcm or len(pcm) % 2:
        raise ValueError("System audio requires complete int16 PCM samples")
    return json.dumps(
        {
            "request_id": request_id,
            "cue": cue,
            "audio_base64": base64.b64encode(pcm).decode("ascii"),
        },
        separators=(",", ":"),
    )


def decode_system_audio(message: str) -> tuple[str, str, bytes]:
    """Return request ID, cue name and PCM, rejecting malformed requests."""
    try:
        payload = json.loads(message)
        request_id = payload["request_id"]
        cue = payload["cue"]
        encoded = payload["audio_base64"]
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("System audio requires a request ID")
        if not isinstance(cue, str) or not cue.strip():
            raise ValueError("System audio requires a cue name")
        if not isinstance(encoded, str):
            raise ValueError("System audio must be base64 text")
        pcm = base64.b64decode(encoded, validate=True)
        if not pcm or len(pcm) % 2:
            raise ValueError("System audio requires complete int16 PCM samples")
        return request_id, cue, pcm
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise ValueError("Invalid system audio request") from exc
