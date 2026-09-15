"""ROS-independent state for one streamed voice-dialogue turn."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass


TTS_CLEAN_RE = re.compile(r"[*#_~`>\[\]\(\)\{\}]")
DEFAULT_PUNCTUATIONS = frozenset({"。", "？", ".", "?", "！", "!"})


@dataclass(frozen=True)
class StreamDecision:
    turn_id: str
    tts_text: str | None = None


@dataclass(frozen=True)
class ReplyDecision:
    turn_id: str
    corrected_text: str
    ai_text: str
    tts_tail: str | None = None


class DialogTurnController:
    """Accumulates streamed LLM text into one correlated output turn."""

    def __init__(self, *, turn_id_factory=None) -> None:
        self._turn_id_factory = turn_id_factory or (lambda: uuid.uuid4().hex[:12])
        self._active_turn_id: str | None = None
        self._sentence_buffer = ""
        self._punc_count = 0
        self._correction_done = False

    def ensure_turn_id(self) -> str:
        if self._active_turn_id is None:
            self._active_turn_id = self._turn_id_factory()
        return self._active_turn_id

    def set_turn_id(self, turn_id: str) -> None:
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("turn_id must be a non-empty string")
        self._active_turn_id = turn_id

    def consume_chunk(self, text: str) -> StreamDecision | None:
        if not text:
            return None

        turn_id = self.ensure_turn_id()
        self._sentence_buffer += text
        if not self._correction_done:
            if "\n" in self._sentence_buffer:
                _, self._sentence_buffer = self._sentence_buffer.split("\n", 1)
                self._correction_done = True
                self._punc_count = self._count_punctuation(self._sentence_buffer)
            elif len(self._sentence_buffer) > 60:
                self._correction_done = True
                self._punc_count = self._count_punctuation(self._sentence_buffer)
            else:
                return StreamDecision(turn_id=turn_id)

        self._punc_count += self._count_punctuation(text)
        if self._punc_count < 2:
            return StreamDecision(turn_id=turn_id)

        tts_text = self._clean_tts(self._sentence_buffer)
        self._sentence_buffer = ""
        self._punc_count = 0
        return StreamDecision(turn_id=turn_id, tts_text=tts_text or None)

    def finish_reply(self, text: str) -> ReplyDecision | None:
        text = text.strip()
        if not text:
            return None

        turn_id = self.ensure_turn_id()
        corrected_text = ""
        ai_text = text
        if text.startswith("you:"):
            lines = text.split("\n", 1)
            corrected_text = lines[0][4:].strip()
            ai_text = lines[1].strip() if len(lines) > 1 else ""
            if ai_text.startswith("ai:"):
                ai_text = ai_text[3:].strip()

        tts_tail = self._clean_tts(self._sentence_buffer)
        self._reset_stream()
        return ReplyDecision(
            turn_id=turn_id,
            corrected_text=corrected_text,
            ai_text=ai_text,
            tts_tail=tts_tail or None,
        )

    def finish(self) -> str:
        turn_id = self.ensure_turn_id()
        self._reset_stream()
        self._active_turn_id = None
        return turn_id

    def _reset_stream(self) -> None:
        self._sentence_buffer = ""
        self._punc_count = 0
        self._correction_done = False

    @staticmethod
    def _clean_tts(text: str) -> str:
        return TTS_CLEAN_RE.sub("", text.strip()).strip()

    @staticmethod
    def _count_punctuation(text: str) -> int:
        return sum(char in DEFAULT_PUNCTUATIONS for char in text)
