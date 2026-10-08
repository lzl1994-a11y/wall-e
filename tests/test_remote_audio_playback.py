"""Exercise the actual ROS playback callbacks without opening a speaker."""
import json
from array import array
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rclpy")
import rclpy
from std_msgs.msg import MultiArrayDimension, String, UInt8MultiArray
from nodes.audio_playback_node import AudioPlaybackNode


@pytest.fixture
def playback():
    rclpy.init()
    with patch("nodes.audio_playback_node.MixingPlaybackService"):
        node = AudioPlaybackNode()
        try:
            yield node
        finally:
            node.destroy_node()
            rclpy.shutdown()


def state(node, action, epoch, mode=None):
    payload = {"state": action, "epoch": epoch}
    if mode:
        payload["mode"] = mode
    node._on_remote_intercom_state(String(data=json.dumps(payload)))


def pcm(node, epoch):
    msg = UInt8MultiArray(data=array("B", np.ones(960, dtype=np.int16).tobytes()))
    msg.layout.dim = [MultiArrayDimension(label=f"walle.remote_audio_epoch:{epoch}")]
    node._on_remote_audio(msg)


def test_stop_rejects_delayed_pcm(playback):
    state(playback, "start", 1)
    pcm(playback, 1)
    playback._player.play_realtime.assert_called_once()
    state(playback, "stop", 1)
    pcm(playback, 1)
    playback._player.stop_speech.assert_called_once()
    playback._player.play_realtime.assert_called_once()


def test_old_pcm_cannot_enter_the_next_ptt_turn(playback):
    state(playback, "start", 1)
    state(playback, "stop", 1)
    state(playback, "start", 2)
    pcm(playback, 1)
    playback._player.play_realtime.assert_not_called()
    pcm(playback, 2)
    playback._player.play_realtime.assert_called_once()
    state(playback, "stop", 1)
    pcm(playback, 2)
    assert playback._player.play_realtime.call_count == 2


def test_playback_defaults_closed_and_rejects_untagged_pcm(playback):
    pcm(playback, 1)
    state(playback, "start", 1)
    playback._on_remote_audio(UInt8MultiArray(data=[0, 0] * 960))
    playback._player.play_realtime.assert_not_called()


def test_call_and_ptt_overlap_do_not_stop_each_other(playback):
    state(playback, "start", 1, "session")
    state(playback, "start", 1)
    state(playback, "stop", 1)
    pcm(playback, 1)
    playback._player.stop_speech.assert_not_called()
    playback._player.play_realtime.assert_called_once()
    state(playback, "end", 1, "session")
    pcm(playback, 1)
    playback._player.stop_speech.assert_called_once()
    playback._player.play_realtime.assert_called_once()
