"""Pure screen-presentation decisions for voice dialogue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.dialog_turn import ReplyDecision


@dataclass(frozen=True)
class DialogPresentationDecision:
    screen_payload: dict[str, Any]
    tts_tail: str | None = None


class DialogPresentationController:
    """Build stable screen payloads while keeping ROS publishing at the edge."""

    @staticmethod
    def wake_listening() -> dict[str, str]:
        return {"page": "chat", "text": "正在听...", "source": "wake_word"}

    @staticmethod
    def timed_out() -> dict[str, str]:
        return {
            "page": "idle",
            "text": "说「瓦力瓦力」唤醒我",
            "source": "timeout",
        }

    @staticmethod
    def reply(
        decision: ReplyDecision,
        action_results: Any,
    ) -> DialogPresentationDecision:
        actions = action_results if isinstance(action_results, list) else []
        return DialogPresentationDecision(
            screen_payload={
                "turn_id": decision.turn_id,
                "corrected_text": decision.corrected_text,
                "ai_text": decision.ai_text,
                "actions": actions,
                "source": "voice_chat",
            },
            tts_tail=decision.tts_tail,
        )


__all__ = ["DialogPresentationController", "DialogPresentationDecision"]
