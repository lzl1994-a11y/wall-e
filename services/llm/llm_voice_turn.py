"""Pure service managing voice turn state, action results normalization, and final decision."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from services.llm.llm_stream_response import (
    DEFAULT_CLAUSE_PUNCTUATIONS,
    DEFAULT_FIRST_TTS_CLAUSE_MIN_CHARS,
    DEFAULT_PUNCTUATIONS,
    FALLBACK_EMPTY_REPLY,
    StreamEventDecision,
    StreamResponseAccumulator,
)

ALLOWED_SUCCESS_STATUSES = frozenset({"completed", "skipped"})


@dataclass(frozen=True)
class VoiceTurnFinalDecision:
    user_text: str
    clean_text: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    needs_empty_answer_retry: bool = False
    should_publish_final_tts: bool = False


class LLMVoiceTurnState:
    """Encapsulates ordinary voice dialog turn lifecycle, action accumulation, and final response policy."""

    FALLBACK_EMPTY_REPLY = FALLBACK_EMPTY_REPLY

    def __init__(
        self,
        user_prompt: str,
        punctuations: Optional[set[str] | frozenset[str]] = None,
        clause_punctuations: Optional[set[str] | frozenset[str]] = None,
        first_tts_clause_min_chars: int = DEFAULT_FIRST_TTS_CLAUSE_MIN_CHARS,
        *,
        accumulator: Optional[StreamResponseAccumulator] = None,
    ) -> None:
        self.user_prompt: str = user_prompt
        if accumulator is not None:
            self.accumulator = accumulator
        else:
            kwargs: dict[str, Any] = {
                "first_tts_clause_min_chars": first_tts_clause_min_chars,
            }
            if punctuations is not None:
                kwargs["punctuations"] = punctuations
            if clause_punctuations is not None:
                kwargs["clause_punctuations"] = clause_punctuations
            self.accumulator = StreamResponseAccumulator(**kwargs)

        self.corrected_text: str = ""
        self.corrected_text_published: bool = False
        self.actions: list[dict[str, Any]] = []
        self.pending_actions: list[dict[str, Any]] = []
        self.action_failure: Optional[dict[str, Any]] = None
        self.expression_published: bool = False

    # ------------------------------------------------------------------
    # Correction and Expression State
    # ------------------------------------------------------------------

    def normalize_corrected_text(self, value: Optional[str]) -> str:
        """Normalize speech correction text, updating internal state without publishing."""
        normalized = (value or self.user_prompt).strip() or self.user_prompt
        self.corrected_text = normalized
        self.corrected_text_published = True
        return normalized

    @property
    def needs_default_expression(self) -> bool:
        """True if the initial default expression has not been published yet."""
        return not self.expression_published

    def mark_expression_published(self) -> None:
        """Mark dialog expression as published so default expression won't re-fire."""
        self.expression_published = True

    # ------------------------------------------------------------------
    # Stream Response Accumulator Delegation
    # ------------------------------------------------------------------

    def process_event(self, event: Any) -> StreamEventDecision:
        """Delegate incoming LLM stream event to internal accumulator."""
        return self.accumulator.process_event(event)

    def record_spoken(self, text: str) -> None:
        """Record spoken text fragment into internal accumulator."""
        self.accumulator.record_spoken(text)

    def record_rejected_action(self, name: str, reason: str) -> None:
        """Record action rejection into internal accumulator."""
        self.accumulator.record_rejected_action(name, reason)

    def take_tail_tts(self) -> str | None:
        """Flush remaining buffered text for speech playback."""
        return self.accumulator.take_tail_tts()

    # ------------------------------------------------------------------
    # Tool Proposals & Actions Lifecycle
    # ------------------------------------------------------------------

    def record_pending_action(
        self,
        turn_id: str,
        name: str,
        arguments: Any,
    ) -> dict[str, Any]:
        """Record a proposed action into pending actions and tool proposals."""
        action_payload = {
            "turn_id": turn_id,
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }
        self.actions.append(action_payload)
        self.pending_actions.append({
            "name": name,
            "arguments": arguments,
        })
        self.accumulator.record_tool_proposal(action_payload)
        return action_payload

    def apply_action_sequence(
        self,
        sequence: Mapping[str, Any],
        turn_id: str,
    ) -> list[dict[str, Any]]:
        """Normalize action execution results and extract the first failure."""
        sequence_results = sequence.get("results", [])
        actions = []
        for result in sequence_results:
            actions.append({
                "turn_id": turn_id,
                "name": result.get("name") or result.get("action", ""),
                "arguments": json.dumps(result.get("arguments", {}), ensure_ascii=False),
                "status": result.get("status", "failed"),
                "request_id": result.get("request_id", ""),
                "reason": result.get("reason", ""),
            })
        self.actions = actions
        self.action_failure = next(
            (
                result
                for result in sequence_results
                if result.get("status") not in ALLOWED_SUCCESS_STATUSES
            ),
            None,
        )
        return actions

    # ------------------------------------------------------------------
    # Final Decision & Retry
    # ------------------------------------------------------------------

    def decide_final_turn(self) -> VoiceTurnFinalDecision:
        """Produce final turn decision determining user text, clean text, and TTS need."""
        user_text = self.corrected_text if self.corrected_text else self.user_prompt
        clean_text = self.accumulator.decide_final_text(
            action_failure=self.action_failure,
            actions=self.actions,
        )
        needs_retry = not bool(clean_text)
        should_publish_final_tts = not bool(self.accumulator.spoken_parts)
        return VoiceTurnFinalDecision(
            user_text=user_text,
            clean_text=clean_text,
            actions=list(self.actions),
            needs_empty_answer_retry=needs_retry,
            should_publish_final_tts=should_publish_final_tts,
        )

    def resolve_final_text(self, retry_text: Optional[str]) -> str:
        """Resolve final text after retry, falling back to empty reply constant if empty."""
        if retry_text and str(retry_text).strip():
            return str(retry_text).strip()
        return self.accumulator.FALLBACK_EMPTY_REPLY
