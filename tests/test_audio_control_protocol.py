from services.audio.audio_control_protocol import (
    decode_audio_control,
    encode_stop_speech_control,
)


def test_stop_speech_control_round_trip():
    assert decode_audio_control(encode_stop_speech_control("remote_barge_in")) == {
        "action": "stop_speech",
        "source": "remote_barge_in",
    }


def test_audio_control_rejects_unknown_fields_and_actions():
    assert decode_audio_control('{"action":"play_file","source":"x"}') is None
    assert decode_audio_control('{"action":"stop_speech","source":"x","extra":1}') is None
