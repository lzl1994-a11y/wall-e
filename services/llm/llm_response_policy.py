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
        "修正文本",
        "纠错文本",
        "校正文本",
        "识别修正",
        "修正后文本",
        "corrected_text",
        "corrected text",
    }
    TTS_CLEAN_RE = re.compile(
        "[^\\w\\s\u4e00-\u9fa5\uff0c\u3002\uff1f\uff01\u3001\uff1a\uff1b\u201c\u201d\uff08\uff09\u300a\u300b.,?!]"
    )
    OUTPUT_LINE_PREFIX_RE = re.compile(
        r"^\s*(?:第二行|最终回答|最终答案|回答|回复)\s*[:：]\s*", re.IGNORECASE
    )
    VISUAL_ANSWER_PREFIXES = (
        "【修正文本】",
        "修正文本:",
        "修正文本：",
        "ai:",
        "AI:",
        "ai：",
        "AI：",
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
        first_line = (first_line or "").strip()
        if not first_line:
            return None

        cleaned = first_line.lstrip(" \t>*-#")
        cleaned = re.sub(
            r"^\s*第一行\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        label = ""
        value = ""
        if cleaned.startswith("【") and "】" in cleaned:
            label, value = cleaned[1:].split("】", 1)
        elif cleaned.startswith("[") and "]" in cleaned:
            label, value = cleaned[1:].split("]", 1)
        else:
            value = cleaned

        if label:
            label = label.strip().lower()
            if label not in cls.CORRECTION_LABELS:
                return None
            value = value.lstrip(" \t:：")
            return value.strip().strip('"“”') or None

        # Fallbacks for plain labeled responses, e.g. corrected_text: hello.
        for sep in (":", "："):
            if sep not in value:
                continue
            maybe_label, maybe_text = value.split(sep, 1)
            if maybe_label.strip().lower() in cls.CORRECTION_LABELS:
                return maybe_text.strip().strip('"“”') or None

        for label in cls.CORRECTION_LABELS:
            if value.lower().startswith(label.lower()):
                remainder = value[len(label):]
                if remainder and remainder[0] not in " \t:：":
                    continue
                maybe_text = remainder.strip(" \t:：")
                return maybe_text.strip().strip('"“”') or None

        return None

    @classmethod
    def is_correction_label_only(cls, text: str) -> bool:
        cleaned = (text or "").strip().lstrip(" \t>*-#")
        cleaned = cleaned.strip("[]【】").strip(" \t:：").lower()
        return cleaned in cls.CORRECTION_LABELS

    @classmethod
    def strip_answer_prefix(cls, text: str) -> str:
        return cls.OUTPUT_LINE_PREFIX_RE.sub("", text or "", count=1)

    @classmethod
    def strip_correction_line(cls, text: str) -> str:
        if "\n" not in (text or ""):
            return text or ""
        first_line, rest = text.split("\n", 1)
        if cls.extract_corrected_text(first_line):
            return cls.strip_answer_prefix(rest)
        if cls.is_correction_label_only(first_line):
            if "\n" not in rest:
                return ""
            _, answer = rest.split("\n", 1)
            return cls.strip_answer_prefix(answer)
        return text

    @classmethod
    def clean_tts_text(cls, text: str) -> str:
        return cls.TTS_CLEAN_RE.sub("", text or "").strip()

    @classmethod
    def sanitize_speech_text(cls, text: str) -> str:
        clean = cls.strip_answer_prefix(cls.strip_correction_line(text)).strip()
        if "\n" not in clean and cls.extract_corrected_text(clean):
            return ""
        return cls.clean_tts_text(clean)

    @classmethod
    def clean_visual_answer(cls, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
        # 防止兼容模型仍然套用本项目普通对话的标签格式。
        for prefix in cls.VISUAL_ANSWER_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" ：:")
        return cls.clean_tts_text(text)


__all__ = ["LLMResponsePolicy"]
