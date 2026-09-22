"""Pure service for preparing and parsing LLM empty-answer retry requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from services.llm.llm_response_policy import LLMResponsePolicy


@dataclass(frozen=True)
class EmptyAnswerRetryRequest:
    prompt: str
    history: list[dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = False
    structured_answer: bool = False
    max_tokens_override: int = 256


def build_empty_answer_retry_prompt(user_prompt: str) -> str:
    """Build the exact prompt used for single empty-answer retry."""
    return (
        f"用户说：{user_prompt}\n"
        "请直接用一到两句简短自然的中文回答。只输出回答正文，不要输出纠错标签、"
        "分析过程、Markdown、动作说明或任何前缀。"
    )


def calculate_retry_max_tokens(settings_or_tokens: Any = None) -> int:
    """Calculate token override for empty-answer retry maintaining existing compatibility semantics."""
    if isinstance(settings_or_tokens, dict):
        configured = settings_or_tokens.get("max_tokens", 0)
    elif isinstance(settings_or_tokens, int):
        configured = settings_or_tokens
    else:
        configured = 0

    retry_tokens = configured if isinstance(configured, int) else 0
    if retry_tokens <= 0:
        retry_tokens = 256
    return min(max(retry_tokens, 128), 256)


def build_empty_answer_retry_request(
    user_prompt: str,
    *,
    history: Optional[Sequence[dict[str, Any]]] = None,
    settings: Any = None,
) -> EmptyAnswerRetryRequest:
    """Construct frozen model request for empty-answer retry."""
    return EmptyAnswerRetryRequest(
        prompt=build_empty_answer_retry_prompt(user_prompt),
        history=list(history) if history is not None else [],
        tools_enabled=False,
        structured_answer=False,
        max_tokens_override=calculate_retry_max_tokens(settings),
    )


def parse_retry_answer(text: Any) -> str:
    """Parse text/chunks returned from empty-answer retry into a sanitized speech string."""
    if isinstance(text, str):
        raw = text.strip()
    elif isinstance(text, (list, tuple)):
        raw = "".join(str(chunk) for chunk in text).strip()
    elif text is None:
        raw = ""
    else:
        raw = str(text).strip()

    if not raw:
        return ""
    if "\n" not in raw and LLMResponsePolicy.extract_corrected_text(raw):
        return ""
    return LLMResponsePolicy.sanitize_speech_text(raw)


class EmptyAnswerRetryAccumulator:
    """Accumulates text stream events and parses final speech response for empty-answer retry."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def process_event(self, event: Any) -> None:
        """Accumulate only type=text events with non-empty string content."""
        if not isinstance(event, Mapping):
            return
        if event.get("type") == "text":
            content = event.get("content")
            if isinstance(content, str) and content:
                self._chunks.append(content)

    def raw_text(self) -> str:
        """Return concatenated raw chunks stripped."""
        return "".join(self._chunks).strip()

    def parse_answer(self) -> str:
        """Parse accumulated chunks according to speech sanitization rules."""
        return parse_retry_answer(self.raw_text())
