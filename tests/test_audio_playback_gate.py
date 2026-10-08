import json
from types import SimpleNamespace

import pytest

from services.remote.audio_playback_gate import AudioPlaybackGate, audio_message_epoch
from services.remote.remote_protocol import encode_voice_state, encode_call_state


def test_stopped_epoch_cannot_be_reopened_and_old_audio_cannot_enter_new_turn():
    gate = AudioPlaybackGate()
    assert not gate.accepts(1)
    assert gate.update(encode_voice_state("start", epoch=1))
    assert gate.accepts(1)
    assert gate.update(encode_voice_state("stop", epoch=1))
    assert not gate.accepts(1)
    assert not gate.update(encode_voice_state("start", epoch=1))
    assert gate.update(encode_voice_state("start", epoch=2))
    assert not gate.accepts(1)
    assert not gate.update(encode_voice_state("stop", epoch=1))
    assert gate.accepts(2)


def test_stop_before_start_is_fail_closed_and_safety_close_needs_a_new_epoch():
    gate = AudioPlaybackGate()
    assert gate.update(encode_voice_state("stop", epoch=1))
    assert not gate.update(encode_voice_state("start", epoch=1))
    assert gate.update(encode_voice_state("start", epoch=2))
    gate.close()
    assert not gate.accepts(2)
    assert not gate.update(encode_voice_state("start", epoch=2))
    assert gate.update(encode_voice_state("start", epoch=3))
    assert gate.accepts(3)


def test_call_and_voice_share_an_epoch_but_keep_independent_lifetimes():
    gate = AudioPlaybackGate()
    gate.update(encode_call_state("start", epoch=1))
    gate.update(encode_voice_state("start", epoch=1))
    gate.update(encode_voice_state("stop", epoch=1))
    assert gate.accepts(1)
    gate.update(encode_call_state("end", epoch=1))
    assert not gate.accepts(1)


@pytest.mark.parametrize("raw", ["invalid", "null", "[]", '{}',
    '{"state":"start","epoch":true}', '{"state":"start","epoch":0}',
    '{"state":"start","epoch":-1}', '{"state":"start","epoch":1,"mode":"unknown"}',
    '{"state":"end","epoch":1}', '{"state":"stop","epoch":1,"mode":"session"}',
    '{"state":"start","epoch":9223372036854775808}'])
def test_malformed_state_never_opens_playback(raw):
    gate = AudioPlaybackGate()
    assert not gate.update(raw)
    assert not gate.active


@pytest.mark.parametrize("labels", [[], ['walle.remote_audio_epoch:1','walle.remote_audio_epoch:2'],
    ['walle.remote_audio_epoch:-1'], ['walle.remote_audio_epoch:0'],
    ['walle.remote_audio_epoch:abc'], ['walle.remote_audio_epoch:١'],
    ['walle.remote_audio_epoch:9223372036854775808']])
def test_malformed_pcm_metadata_is_rejected(labels):
    msg = SimpleNamespace(layout=SimpleNamespace(dim=[SimpleNamespace(label=v) for v in labels]))
    assert audio_message_epoch(msg) is None


def test_valid_pcm_metadata_and_legacy_state_encoding():
    msg = SimpleNamespace(layout=SimpleNamespace(dim=[SimpleNamespace(label='walle.remote_audio_epoch:123')]))
    assert audio_message_epoch(msg) == 123
    assert json.loads(encode_voice_state("start")) == {"state":"start"}
    assert json.loads(encode_call_state("start")) == {"state":"start","mode":"session"}
