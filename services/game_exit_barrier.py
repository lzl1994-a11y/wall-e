"""Pure exit barrier state coordinating session completion and trailing audio completion."""

from __future__ import annotations


class GameExitBarrier:
    """Pure state model coordinating game session termination and trailing audio completion."""

    def __init__(self) -> None:
        self._awaiting_audio_end: bool = False
        self._session_finished: bool = False
        self._audio_finished: bool = False

    def prepare_audio_end(self) -> None:
        """Indicate trailing audio has been queued and begin awaiting audio completion."""
        self._awaiting_audio_end = True
        self._audio_finished = False

    def mark_session_finished(self) -> bool:
        """Record session finished. Return True if exit may complete immediately."""
        self._session_finished = True
        if not self._awaiting_audio_end:
            return True
        if self._audio_finished:
            return True
        return False

    def mark_audio_finished(self) -> bool:
        """Record audio finished if awaiting. Return True if exit may complete."""
        if not self._awaiting_audio_end:
            return False
        self._audio_finished = True
        if self._session_finished:
            return True
        return False

    def reset(self) -> None:
        """Reset session and audio waiting states for next game round."""
        self._awaiting_audio_end = False
        self._session_finished = False
        self._audio_finished = False
