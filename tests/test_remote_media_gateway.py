"""Integration coverage on the ROS/aiortc host; cleanly skipped elsewhere."""
import json
import asyncio
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("aiortc")
from nodes.remote_webrtc_gateway_node import CameraVideoTrack, RobotAudioTrack, RemoteWebRtcGateway
from services.remote.realtime_control import ChannelSequenceGate


def gateway():
    value = RemoteWebRtcGateway.__new__(RemoteWebRtcGateway)
    value.node = Mock()
    value._sequence_gate = ChannelSequenceGate()
    value._publish_control = Mock()
    value._set_remote_voice = Mock()
    value._stop_robot_outputs = Mock()
    value._schedule_peer_close = Mock()
    value._send_event = Mock()
    value._video_track = Mock()
    return value


def control(seq):
    return json.dumps({"type": "control", "seq": seq,
                       "vector": dict(forward=0, turn=0, yaw=0, pitch=0)})


def test_reordered_motion_does_not_break_voice_or_disconnect():
    value = gateway()
    value._on_data_message(control(10), "motion")
    value._on_data_message(control(9), "motion")
    value._on_data_message('{"type":"voice","seq":1,"state":"start"}', "control")
    value._publish_control.assert_called_once()
    value._set_remote_voice.assert_called_once_with(True)
    value._schedule_peer_close.assert_not_called()


def test_emergency_stop_blocks_in_flight_motion_before_async_close():
    value = gateway()
    value._on_data_message('{"type":"stop","seq":1}', "control")
    value._on_data_message(control(99), "motion")
    value._publish_control.assert_not_called()
    value._stop_robot_outputs.assert_called_once()
    value._schedule_peer_close.assert_called_once()


def test_low_video_profile_is_applied_and_placeholder_keeps_dimensions():
    value = gateway()
    value._on_data_message('{"type":"media","seq":1,"video":"low"}')
    assert value._video_track.profile == "low"
    frame = CameraVideoTrack._make_frame(b"", 320, 240)
    assert (frame.width, frame.height) == (320, 240)


def test_media_timestamps_follow_elapsed_time_after_cpu_stall():
    async def scenario():
        with patch("nodes.remote_webrtc_gateway_node.time.monotonic", return_value=100):
            audio = RobotAudioTrack()
            video = CameraVideoTrack(Mock(latest_camera_frame=lambda: b""))
        with patch("nodes.remote_webrtc_gateway_node.time.monotonic", return_value=102):
            audio_frame = await audio.recv()
            video_frame = await video.recv()
        assert audio_frame.pts == 96000
        assert video_frame.pts == 180000
        audio.stop(); video.stop()
    asyncio.run(scenario())
