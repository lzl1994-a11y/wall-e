import json

from services.remote.remote_protocol import action_request_for, decode_remote_message


def test_decode_remote_control_clamps_protocol_shape_without_mutating_values():
    message = decode_remote_message(json.dumps({
        "type": "control",
        "seq": 4,
        "vector": {"forward": 1, "turn": -0.5, "yaw": 0, "pitch": 0.25},
    }))

    assert message == {
        "type": "control",
        "seq": 4,
        "vector": {"forward": 1.0, "turn": -0.5, "yaw": 0.0, "pitch": 0.25},
    }


def test_decode_remote_message_rejects_invalid_or_unsafe_inputs():
    invalid = [
        {"type": "control", "seq": 1, "vector": {"forward": 2, "turn": 0, "yaw": 0, "pitch": 0}},
        {"type": "control", "seq": 1, "vector": {"forward": 10**1000, "turn": 0, "yaw": 0, "pitch": 0}},
        {"type": "action", "seq": 1, "name": "shell_command"},
        {"type": "voice", "seq": 1, "state": "toggle"},
        {"type": "stop", "seq": True, "reason": "operator"},
        {"type": "call", "seq": 1, "state": "toggle"},
        {"type": "control", "seq": 1, "vector": {"forward": 0, "turn": 0, "yaw": 0, "pitch": 0}, "extra": 1},
    ]

    assert all(decode_remote_message(json.dumps(value)) is None for value in invalid)


def test_call_lifecycle_is_distinct_from_legacy_push_to_talk():
    assert decode_remote_message(
        json.dumps({"type": "call", "seq": 9, "state": "start"})
    ) == {"type": "call", "seq": 9, "state": "start"}
    assert decode_remote_message(
        json.dumps({"type": "voice", "seq": 10, "state": "start"})
    ) == {"type": "voice", "seq": 10, "state": "start"}


def test_protocol_rejects_non_finite_and_oversized_messages():
    assert decode_remote_message(
        '{"type":"control","seq":1,"vector":{"forward":NaN,"turn":0,"yaw":0,"pitch":0}}'
    ) is None
    assert decode_remote_message("x" * (64 * 1024 + 1)) is None


def test_browser_actions_use_the_existing_sequence_skill():
    assert action_request_for("wave_hello") == (
        "play_sequence",
        {"sequence_name": "wave_hello"},
    )


def test_media_profile_is_bounded_and_allowlisted():
    for profile in ("normal", "low"):
        message = {"type": "media", "seq": 1, "video": profile}
        assert decode_remote_message(json.dumps(message)) == message
    for profile in ("4k", None, 5, {}, ["low"]):
        assert decode_remote_message(json.dumps({"type": "media", "seq": 1, "video": profile})) is None


def test_manual_video_accepts_only_source_supported_sizes_and_frame_rates():
    for width, height in [(320, 240), (480, 360), (640, 480)]:
        for fps in (5, 10, 15):
            message = dict(type="media", seq=7, mode="manual", width=width, height=height, fps=fps)
            assert decode_remote_message(json.dumps(message)) == message
    message = dict(type="media", seq=7, mode="manual", width=480, height=360, fps=15)
    for key, value in [("width", 1920), ("height", 480), ("fps", 30), ("fps", 0), ("fps", True),
                       ("fps", 15.0), ("width", []), ("mode", "unknown")]:
        assert decode_remote_message(json.dumps({**message, key: value})) is None
    assert decode_remote_message(json.dumps({**message, "extra": 1})) is None
    assert decode_remote_message(json.dumps({key: value for key, value in message.items() if key != "fps"})) is None


def test_explicit_automatic_video_and_legacy_profiles_are_compatible():
    for profile in ("low", "normal"):
        message = dict(type="media", seq=7, mode="auto", video=profile)
        assert decode_remote_message(json.dumps(message)) == message
    assert decode_remote_message(json.dumps(dict(type="media", seq=7, mode="auto", width=480, height=360, fps=15))) is None
