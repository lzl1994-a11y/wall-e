"""Pure response-length and speech-text policy for the text LLM node."""

from __future__ import annotations

import re


class LLMResponsePolicy:
    LONG_FORM_REQUEST_RE = re.compile(
        r"(?:背(?:诵)?|朗(?:诵|读)|念|读)(?:一下|一遍|给我听)?|"
        r"全文|完整(?:版|内容)?|全部|整首|从头到尾"
    )
    LONG_FORM_MAX_TOKENS = 2048
    CORRECTION_LABELS = {
        "修正文本", "纠错文本", "校正文本", "识别修正", "修正后文本",
        "corrected_text", "corrected text",
    }
    TTS_CLEAN_RE = re.compile(
        "[^\\w\\s\u4e00-\u9fa5，。？！、】【：；“”（）《》.,?!]"
    )
    OUTPUT_LINE_PREFIX_RE = re.compile(
        r"^\s*(?:第二行|最终回答|最终答案|回答|回复)\s*[:：]\s*", re.IGNORECASE
    )

    @classmethod
    def is_long_form_request(cls, user_prompt: str) -> bool:
        return bool(cls.LONG_FORM_REQUEST_RE.search(user_prompt or ""))

    @classmethod
    def max_tokens(cls, configured_tokens: object, *, long_form: bool) -> int | None:
        if not long_form:
            return None
        configured = configured_tokens if isinstance(configured_tokens, int) else 0
        return max(cls.LONG_FORM_MAX_TOKENS, configured)

    @classmethod
    def extract_corrected_text(cls, first_line: str) -> str | None:
        cleaned = (first_line or "").strip().lstrip(" \t>*-#")
        if not cleaned:
            return None
        cleaned = re.sub(r"^\s*第一行\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        label = ""
        value = cleaned
        if cleaned.startswith("【") and "】" in cleaned:
            label, value = cleaned[1:].split("】", 1)
        elif cleaned.startswith("[") and "]" in cleaned:
            label, value = cleaned[1:].split("]", 1)
        if label:
            if label.strip().lower() not in cls.CORRECTION_LABELS:
                return None
            return value.lstrip(" \t:：").strip().strip('"“”') or None
        for separator in (":", "："):
            if separator in value:
                possible, text = value.split(separator, 1)
                if possible.strip().lower() in cls.CORRECTION_LABELS:
                    return text.strip().strip('"“”') or None
        for label in cls.CORRECTION_LABELS:
            if value.lower().startswith(label.lower()):
                remainder = value[len(label):]
                if not remainder or remainder[0] in " \t:：":
                    return remainder.strip(" \t:：").strip('"“”') or None
        return None

    @classmethod
    def sanitize_speech_text(cls, text: str) -> str:
        text = cls.strip_correction_line(text)
        text = cls.OUTPUT_LINE_PREFIX_RE.sub("", text or "", count=1).strip()
        if "\n" not in text and cls.extract_corrected_text(text):
            return ""
        return cls.TTS_CLEAN_RE.sub("", text).strip()

    @classmethod
    def strip_correction_line(cls, text: str) -> str:
        if "\n" not in text:
            return text
        first, rest = text.split("\n", 1)
        if cls.extract_corrected_text(first):
            return cls.OUTPUT_LINE_PREFIX_RE.sub("", rest, count=1)
        label = first.strip().lstrip(" \t>*-#").strip("[]【】").strip(" \t:：").lower()
        if label in cls.CORRECTION_LABELS:
            return rest.split("\n", 1)[1] if "\n" in rest else ""
        return text


__all__ = ["LLMResponsePolicy"]
