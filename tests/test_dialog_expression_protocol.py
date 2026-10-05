import json

from services.dialog.dialog_expression_protocol import (
    decode_dialog_expression,
    encode_dialog_expression,
    expression_to_pose_expression,
    expression_to_tft_mood,
    split_dialog_expression_prefix,
)


def test_shared_expression_maps_to_tft_moods():
    assert expression_to_tft_mood("happy") == "heart"
    assert expression_to_tft_mood("angry") == "flame"
    assert expression_to_tft_mood("disdain") == "flame"
    assert expression_to_tft_mood("thinking") == "dot"


def test_expression_alias_is_normalized_at_protocol_boundary():
    payload = encode_dialog_expression("love", "medium", "turn-1")
    assert decode_dialog_expression(payload) == {
        "expression": "happy",
        "intensity": "medium",
        "turn_id": "turn-1",
    }


def test_action_only_expressions_reuse_existing_dialog_poses():
    assert expression_to_pose_expression("curious") == "thinking"
    assert expression_to_pose_expression("disdain") == "confused"
    assert expression_to_pose_expression("angry") == "concerned"


def test_dialog_expression_prefix_is_removed_before_tts():
    assert split_dialog_expression_prefix("😠别拆我。") == (
        "angry", "medium", "别拆我。", True
    )
    assert split_dialog_expression_prefix("  😍 谢谢你！") == (
        "happy", "medium", "谢谢你！", True
    )


def test_missing_dialog_expression_prefix_falls_back_without_losing_text():
    assert split_dialog_expression_prefix("你好。") == (
        "neutral", "low", "你好。", False
    )
