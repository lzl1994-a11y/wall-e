"""Validated ROS contract for semantic conversational expressions."""

from __future__ import annotations

import json


DIALOG_EXPRESSION_TOPIC = "dialog_expression"
DIALOG_EXPRESSION_TARGET_TOPIC = "/servo_targets/dialog_expression"
EXPRESSIONS = frozenset({
    "neutral", "listening", "thinking", "happy",
    "sad", "surprised", "confused", "concerned",
})
INTENSITIES = frozenset({"low", "medium", "high"})


def normalize_expression(expression, intensity):
    expression = str(expression or "").strip().lower()
    intensity = str(intensity or "").strip().lower()
    if expression not in EXPRESSIONS:
        expression = "neutral"
    if intensity not in INTENSITIES:
        intensity = "low"
    return expression, intensity


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
