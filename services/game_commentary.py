"""ROS-independent scheduling state for periodic game-screen commentary."""

from __future__ import annotations

import random
import threading
import time
import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameModeDecision:
    pause_voice: bool = False
    resume_voice: bool = False
    clear_frame: bool = False


@dataclass(frozen=True)
class GameCommentaryResult:
    answer: str = ""
    error: str | None = None


GAME_COMMENTARY_PROMPT = (
    "观察当前 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
    "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。"
)


class GameCommentaryWorkflow:
    """Prepare one game-screen LLM request and normalize its spoken reply."""

    def __init__(self, *, analyze: Callable[[str, str], Any], clean: Callable[[str], str]):
        self._analyze = analyze
        self._clean = clean

    def invoke(self, jpeg: bytes) -> GameCommentaryResult:
        try:
            answer = self._clean(str(self._analyze(
                GAME_COMMENTARY_PROMPT,
                base64.b64encode(jpeg).decode("ascii"),
            ) or ""))
            return GameCommentaryResult(answer=answer)
        except Exception as exc:
            return GameCommentaryResult(error=str(exc))


class GameCommentaryController:
    """Own game-mode transitions, frame retention, and commentary cadence."""

    def __init__(
        self,
        *,
        interval_seconds: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._interval_seconds = interval_seconds or (
            lambda: random.uniform(50.0, 120.0)
        )
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._mode = "robot"
        self._latest_frame: Any = None
        self._next_commentary: float | None = None
        self._commentary_running = False

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> GameModeDecision:
        with self._lock:
            previous = self._mode
            self._mode = mode
            if mode != "robot":
                if mode == "playing" and previous != "playing":
                    self._schedule_next_locked()
                return GameModeDecision(pause_voice=previous == "robot")
            if previous == "robot":
                return GameModeDecision()
            self._latest_frame = None
            self._next_commentary = None
            return GameModeDecision(resume_voice=True, clear_frame=True)

    def accept_frame(self, frame: Any) -> bool:
        with self._lock:
            if self._mode == "robot":
                return False
            self._latest_frame = frame
            return True

    def take_due_frame(self) -> Any | None:
        """Reserve the latest frame for one commentary when its timer is due."""
        with self._lock:
            if self._mode != "playing" or self._commentary_running:
                return None
            now = self._clock()
            if self._next_commentary is None:
                self._schedule_next_locked(now)
                return None
            if now < self._next_commentary:
                return None
            frame = self._latest_frame
            self._schedule_next_locked(now)
            if frame is None:
                return None
            self._commentary_running = True
            return frame

    def can_publish_commentary(self) -> bool:
        with self._lock:
            return self._mode == "playing"

    def finish_commentary(self) -> None:
        with self._lock:
            self._commentary_running = False

    def _schedule_next_locked(self, now: float | None = None) -> None:
        self._next_commentary = (self._clock() if now is None else now) + max(
            0.0, float(self._interval_seconds())
        )


__all__ = [
    "GAME_COMMENTARY_PROMPT",
    "GameCommentaryController",
    "GameCommentaryResult",
    "GameCommentaryWorkflow",
    "GameModeDecision",
]
