"""Validated ROS contract for semantic conversational expressions."""

from __future__ import annotations

import json


DIALOG_EXPRESSION_TOPIC = "dialog_expression"
DIALOG_EXPRESSION_TARGET_TOPIC = "/servo_targets/dialog_expression"
EXPRESSIONS = frozenset({
    "neutral", "listening", "thinking", "happy",
    "sad", "surprised", "confused", "concerned",
    "curious", "disdain", "angry",
})
INTENSITIES = frozenset({"low", "medium", "high"})
TFT_MOODS = frozenset({"dot", "flame", "heart"})

# Ordinary ASR replies use the same one-prefix contract as Xiaozhi: the model
# selects one closed-set marker and the local pipeline removes it before TTS.
# The semantic expression drives the existing dialog pose controller; only
# happy and angry have dedicated TFT eye moods.
DIALOG_EXPRESSION_PREFIXES = {
    "😶": ("neutral", "low"),
    "👂": ("listening", "medium"),
    "🤔": ("thinking", "medium"),
    "😍": ("happy", "medium"),
    "😢": ("sad", "medium"),
    "😮": ("surprised", "high"),
    "😕": ("confused", "medium"),
    "😟": ("concerned", "medium"),
    "🧐": ("curious", "medium"),
    "😒": ("disdain", "medium"),
    "😠": ("angry", "medium"),
}

_EXPRESSION_ALIASES = {
    "calm": "neutral",
    "love": "happy",
    "loving": "happy",
    "anger": "angry",
    "mad": "angry",
}

_POSE_EXPRESSION_ALIASES = {
    "curious": "thinking",
    "disdain": "confused",
    "angry": "concerned",
}


def expression_to_tft_mood(expression):
    """Map the shared semantic expression onto the TFT's three eye moods."""
    value = str(expression or "").strip().lower()
    value = _EXPRESSION_ALIASES.get(value, value)
    if value in {"happy"}:
        return "heart"
    if value == "angry":
        return "flame"
    return "dot"


def expression_to_pose_expression(expression):
    """Reuse the closest configured conversational pose for added semantics."""
    value = str(expression or "").strip().lower()
    value = _EXPRESSION_ALIASES.get(value, value)
    return _POSE_EXPRESSION_ALIASES.get(value, value)


def normalize_expression(expression, intensity):
    expression = str(expression or "").strip().lower()
    expression = _EXPRESSION_ALIASES.get(expression, expression)
    intensity = str(intensity or "").strip().lower()
    if expression not in EXPRESSIONS:
        expression = "neutral"
    if intensity not in INTENSITIES:
        intensity = "low"
    return expression, intensity


def _strip_dialog_expression_label(text):
    """Remove a model-written English label after a valid Emoji prefix."""
    value = str(text or "").lstrip()
    folded = value.lower()
    for label in sorted(EXPRESSIONS, key=len, reverse=True):
        if not folded.startswith(label):
            continue
        suffix = value[len(label):]
        if suffix and suffix[0] not in " \t\r\n:：-—":
            continue
        return suffix.lstrip(" \t\r\n:：-—")
    return value


def split_dialog_expression_prefix(text):
    """Extract one leading semantic marker and return TTS-safe text.

    Missing or unknown markers fail softly to ``neutral/low`` so providers
    that ignore the prompt still produce audible replies.
    """
    value = str(text or "")
    candidate = value.lstrip()
    if candidate:
        mapping = DIALOG_EXPRESSION_PREFIXES.get(candidate[0])
        if mapping is not None:
            expression, intensity = mapping
            return expression, intensity, _strip_dialog_expression_label(
                candidate[1:]
            ), True
    return "neutral", "low", value, False


def encode_dialog_expression(expression, intensity="low", turn_id=""):
    expression, intensity = normalize_expression(expression, intensity)
    return json.dumps({
        "expression": expression,
        "intensity": intensity,
        "turn_id": str(turn_id or ""),
    }, ensure_ascii=False, separators=(",", ":"))


def decode_dialog_expression(payload):
    try:
        value = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    expression = value.get("expression")
    intensity = value.get("intensity")
    if expression not in EXPRESSIONS or intensity not in INTENSITIES:
        return None
    return {
        "expression": expression,
        "intensity": intensity,
        "turn_id": str(value.get("turn_id") or ""),
    }
