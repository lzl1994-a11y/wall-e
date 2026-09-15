"""ROS-independent playback and capture-resume state for voice dialogue."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputDecision:
    accepted: bool
    schedule_capture_resume: bool = False


class DialogOutputController:
    """Coordinates wake and TTS completion before microphone capture resumes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake_response_active = False
        self._wake_request_id: str | None = None
        self._awaiting_tts_playback = False
        self._shutting_down = False

    def start_wake(self, request_id: str) -> OutputDecision:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        with self._lock:
            if self._shutting_down or self._wake_response_active:
                return OutputDecision(accepted=False)
            self._wake_response_active = True
            self._wake_request_id = request_id
            return OutputDecision(accepted=True)

    def finish_wake(self, request_id: str) -> OutputDecision:
        with self._lock:
            if (
                request_id != self._wake_request_id
                or not self._wake_response_active
            ):
                return OutputDecision(accepted=False)
            self._wake_request_id = None
            self._wake_response_active = False
            return OutputDecision(
                accepted=True,
                schedule_capture_resume=(
                    not self._awaiting_tts_playback and not self._shutting_down
                ),
            )

    def wake_is_active(self, request_id: str) -> bool:
        with self._lock:
            return (
                not self._shutting_down
                and self._wake_response_active
                and request_id == self._wake_request_id
            )

    def mark_tts_queued(self) -> None:
        with self._lock:
            self._awaiting_tts_playback = True

    def finish_tts_playback(self) -> OutputDecision:
        with self._lock:
            if not self._awaiting_tts_playback:
                return OutputDecision(accepted=False)
            self._awaiting_tts_playback = False
            return OutputDecision(
                accepted=True,
                schedule_capture_resume=(
                    not self._wake_response_active and not self._shutting_down
                ),
            )

    def can_resume_capture(self) -> bool:
        with self._lock:
            return not (
                self._shutting_down
                or self._wake_response_active
                or self._awaiting_tts_playback
            )

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    @property
    def wake_response_active(self) -> bool:
        with self._lock:
            return self._wake_response_active

    @property
    def wake_request_id(self) -> str | None:
        with self._lock:
            return self._wake_request_id

    @property
    def awaiting_tts_playback(self) -> bool:
        with self._lock:
            return self._awaiting_tts_playback
