"""Correlated wake-response PCM requests for the shared speaker owner."""

from __future__ import annotations

import base64
import binascii
import json


WAKE_AUDIO_TOPIC = "wake_audio_output"
WAKE_AUDIO_DONE_TOPIC = "wake_audio_done"


def encode_wake_audio(request_id: str, pcm: bytes) -> str:
    """Encode one complete 48 kHz, mono, signed-16-bit wake response."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("Wake audio requires a request ID")
    if not pcm or len(pcm) % 2:
        raise ValueError("Wake audio requires complete int16 PCM samples")
    return json.dumps(
        {
            "request_id": request_id,
            "audio_base64": base64.b64encode(pcm).decode("ascii"),
        },
        separators=(",", ":"),
    )


def decode_wake_audio(message: str) -> tuple[str, bytes]:
    """Return the request ID and PCM, or reject a malformed request."""
    try:
        payload = json.loads(message)
        request_id = payload["request_id"]
        encoded = payload["audio_base64"]
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("Wake audio requires a request ID")
        if not isinstance(encoded, str):
            raise ValueError("Wake audio must be base64 text")
        pcm = base64.b64decode(encoded, validate=True)
        if not pcm or len(pcm) % 2:
            raise ValueError("Wake audio requires complete int16 PCM samples")
        return request_id, pcm
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise ValueError("Invalid wake audio request") from exc
