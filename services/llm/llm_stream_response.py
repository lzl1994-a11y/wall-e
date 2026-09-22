"""Pure service for accumulating LLM stream events and deciding dialog speech/text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from services.action.action_acknowledgement import action_acknowledgement
from services.llm.llm_response_policy import LLMResponsePolicy


DEFAULT_PUNCTUATIONS = frozenset({"。", "？", ".", "?", "！", "!"})
DEFAULT_CLAUSE_PUNCTUATIONS = frozenset({"，", ",", "；", ";", "：", ":"})
DEFAULT_FIRST_TTS_CLAUSE_MIN_CHARS = 10

ACTION_FAILURE_REPLY = "这个动作没有确认完成，我已停止后续动作。"
FALLBACK_EMPTY_REPLY = "我刚才卡住了，等我缓一下。"


@dataclass
class StreamEventDecision:
    """Decision output for a single processed LLM stream event."""

    tts_sentences: list[str] = field(default_factory=list)
    expression: dict[str, Any] | None = None
    tool_call: dict[str, Any] | None = None
    finish_reason: str | None = None


class StreamResponseAccumulator:
    """Accumulates LLM stream chunks and decides sentence/text playback policies."""

    ACTION_FAILURE_REPLY = ACTION_FAILURE_REPLY
    FALLBACK_EMPTY_REPLY = FALLBACK_EMPTY_REPLY

    def __init__(
        self,
        *,
        punctuations: set[str] | frozenset[str] = DEFAULT_PUNCTUATIONS,
        clause_punctuations: set[str] | frozenset[str] = DEFAULT_CLAUSE_PUNCTUATIONS,
        first_tts_clause_min_chars: int = DEFAULT_FIRST_TTS_CLAUSE_MIN_CHARS,
    ) -> None:
        self.punctuations = set(punctuations)
        self.clause_punctuations = set(clause_punctuations)
        self.first_tts_clause_min_chars = first_tts_clause_min_chars

        self.text_buffer = ""
        self.sentence_buffer = ""
        self.spoken_parts: list[str] = []
        self._has_spoken = False

        self.tool_proposals: list[dict[str, Any]] = []
        self.rejected_actions: list[tuple[str, str]] = []
        self.finish_reason: str | None = None

    @property
    def has_spoken(self) -> bool:
        return bool(self.spoken_parts) or self._has_spoken

    def record_spoken(self, text: str) -> None:
        if text:
            self.spoken_parts.append(text)
            self._has_spoken = True

    def record_rejected_action(self, name: str, reason: str) -> None:
        self.rejected_actions.append((name, reason))

    def record_tool_proposal(self, tool_call: dict[str, Any]) -> None:
        self.tool_proposals.append(tool_call)

    def process_event(self, event: dict[str, Any]) -> StreamEventDecision:
        """Process one stream event and return any immediate decisions (TTS, expression, tool)."""
        data_type = event.get("type")
        tts_sentences: list[str] = []
        expression: dict[str, Any] | None = None
        tool_call: dict[str, Any] | None = None
        finish_reason: str | None = None

        if data_type == "text":
            chunk = event.get("content", "")
            tts_sentences = self.append_text(chunk)
        elif data_type == "dialog_expression":
            expression = {
                "expression": event.get("expression"),
                "intensity": event.get("intensity"),
            }
        elif data_type == "tool_call":
            tool_call = {
                "name": event.get("name"),
                "arguments": event.get("arguments"),
            }
        elif data_type == "done":
            finish_reason = event.get("finish_reason") or "unknown"
            self.finish_reason = finish_reason

        return StreamEventDecision(
            tts_sentences=tts_sentences,
            expression=expression,
            tool_call=tool_call,
            finish_reason=finish_reason,
        )

    def append_text(self, chunk: str) -> list[str]:
        """Feed a text chunk, accumulating text and returning any boundary-triggered TTS sentences."""
        if not chunk:
            return []
        self.text_buffer += chunk
        ready_sentences: list[str] = []
        for char in chunk:
            self.sentence_buffer += char
            sentence_boundary = char in self.punctuations
            first_clause_boundary = (
                not self.has_spoken
                and char in self.clause_punctuations
                and len(LLMResponsePolicy.clean_tts_text(self.sentence_buffer))
                >= self.first_tts_clause_min_chars
            )
            if sentence_boundary or first_clause_boundary:
                clean_sentence = self.sentence_buffer.strip()
                # Preserve the legacy streaming behavior: correction metadata
                # is handled only when the final reply is assembled, while
                # partial chunks receive character-level TTS filtering.
                tts_safe = LLMResponsePolicy.clean_tts_text(clean_sentence)
                if tts_safe.strip(" .,?!。，？！"):
                    ready_sentences.append(tts_safe)
                    self._has_spoken = True
                self.sentence_buffer = ""
        return ready_sentences

    def take_tail_tts(self) -> str | None:
        """Extract and clear any remaining sentence buffer text as a final TTS sentence."""
        clean_tail = self.sentence_buffer.strip()
        self.sentence_buffer = ""
        if clean_tail:
            tts_safe_tail = LLMResponsePolicy.clean_tts_text(clean_tail)
            if tts_safe_tail.strip(" .,?!。，？！"):
                return tts_safe_tail
        return None

    @classmethod
    def rejected_action_reply(
        cls, rejected_actions: Sequence[tuple[str, str]]
    ) -> str:
        names = {name for name, _reason in rejected_actions}
        if "inspect_camera" in names:
            return "你是想让我打开摄像头看一下吗？"
        return "我不太确定你是不是要我执行这个动作，可以再明确说一下吗？"

    def decide_final_text(
        self,
        *,
        action_failure: bool | object = False,
        actions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Derive the final dialog reply based on accumulated text, actions, and rejections."""
        clean_text = LLMResponsePolicy.sanitize_speech_text(self.text_buffer)
        if action_failure:
            clean_text = self.ACTION_FAILURE_REPLY
        elif self.rejected_actions:
            clean_text = self.rejected_action_reply(self.rejected_actions)

        if "\n" not in self.text_buffer and LLMResponsePolicy.extract_corrected_text(
            self.text_buffer
        ):
            clean_text = ""

        if not clean_text and self.spoken_parts:
            clean_text = "".join(self.spoken_parts).strip()

        effective_actions = actions if actions is not None else self.tool_proposals
        if not clean_text and effective_actions:
            clean_text = action_acknowledgement(effective_actions)

        if not clean_text and self.rejected_actions:
            clean_text = self.rejected_action_reply(self.rejected_actions)

        return clean_text


LLMStreamResponseService = StreamResponseAccumulator
LLMStreamResponse = StreamResponseAccumulator

__all__ = [
    "ACTION_FAILURE_REPLY",
    "DEFAULT_CLAUSE_PUNCTUATIONS",
    "DEFAULT_FIRST_TTS_CLAUSE_MIN_CHARS",
    "DEFAULT_PUNCTUATIONS",
    "FALLBACK_EMPTY_REPLY",
    "LLMStreamResponse",
    "LLMStreamResponseService",
    "StreamEventDecision",
    "StreamResponseAccumulator",
]
