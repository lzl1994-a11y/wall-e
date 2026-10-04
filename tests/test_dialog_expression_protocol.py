import json

from services.dialog.dialog_expression_protocol import (
    decode_dialog_expression,
    encode_dialog_expression,
    expression_to_pose_expression,
    expression_to_tft_mood,
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
