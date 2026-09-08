"""Map sanitized ESP32 provisioning states to pre-generated audio prompts."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time
import wave

from services.audio_output import (
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_WIDTH,
)


ESP32_NETCFG_STATUS_TOPIC = "/esp32_netcfg_status"
NETWORK_PROMPT_ASSETS = {
    "configuring": "esp_network_configuring.wav",
    "connected": "esp_network_connected.wav",
}


def load_prompt_pcm(path: Path) -> bytes:
    """Load and strictly validate the format expected by the shared mixer."""
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != OUTPUT_CHANNELS
            or source.getsampwidth() != OUTPUT_SAMPLE_WIDTH
            or source.getframerate() != OUTPUT_SAMPLE_RATE
            or source.getcomptype() != "NONE"
        ):
            raise ValueError(
                f"System prompt must be {OUTPUT_SAMPLE_RATE} Hz, "
                f"{OUTPUT_CHANNELS}-channel, {OUTPUT_SAMPLE_WIDTH * 8}-bit PCM: {path}"
            )
        pcm = source.readframes(source.getnframes())
    if not pcm:
        raise ValueError(f"System prompt is empty: {path}")
    return pcm


class Esp32NetworkPromptSelector:
    """Deduplicate status events and return the matching immutable PCM clip."""

    MAX_REPLAY_AGE_SECONDS = 30.0

    def __init__(self, asset_directory: Path | str):
        root = Path(asset_directory)
        self._clips = {
            state: load_prompt_pcm(root / filename)
            for state, filename in NETWORK_PROMPT_ASSETS.items()
        }
        self._last_unidentified_state: str | None = None
        self._seen_order: deque[tuple[str, str]] = deque()
        self._seen: set[tuple[str, str]] = set()

    def select(self, message: str) -> tuple[str, str, bytes] | None:
        """Return ``(request_id, cue, pcm)`` for a new audible state."""
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        published_at = payload.get("published_at")
        if isinstance(published_at, (int, float)):
            age = time.time() - float(published_at)
            if age > self.MAX_REPLAY_AGE_SECONDS:
                return None
        state = payload.get("state")
        if not isinstance(state, str):
            return None
        state = state.strip().lower()
        request_id = payload.get("request_id")
        request_id = request_id.strip() if isinstance(request_id, str) else ""

        if request_id:
            key = (request_id, state)
            if key in self._seen:
                return None
            self._remember(key)
        else:
            if state == self._last_unidentified_state:
                return None
            self._last_unidentified_state = state

        pcm = self._clips.get(state)
        if pcm is None:
            return None
        correlation = request_id or f"esp32-netcfg-{state}"
        return correlation, f"esp32_net_{state}", pcm

    def _remember(self, key: tuple[str, str]) -> None:
        self._seen.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > 64:
            self._seen.discard(self._seen_order.popleft())
