"""Integration coverage on the ROS/aiortc host; cleanly skipped elsewhere."""
import json
import asyncio
import os
import time
import threading
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("aiortc")
from aiortc import RTCIceServer
from aiortc.rtcicetransport import connection_kwargs
from PIL import Image, JpegImagePlugin
from nodes.remote_webrtc_gateway_node import (
    CameraVideoTrack, RobotAudioTrack, RemoteWebRtcGateway, _ice_servers_for_transport,
    _ice_servers, _remote_config,
)
from services.remote.realtime_control import ChannelSequenceGate


def test_hardware_answer_selects_h264_when_browser_offers_vp8_first():
    """Exercise the gateway's real answer path, including aiortc negotiation."""
    from aiortc import RTCPeerConnection, RTCConfiguration, VideoStreamTrack
    from aiortc.sdp import SessionDescription

    async def scenario():
        browser = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        browser.addTransceiver("video", direction="recvonly")
        browser.addTransceiver("audio", direction="sendrecv")
        value = gateway()
        value._peer = None
        value._controller_peer_id = None
        value._ice_servers = []
        value._video_backend = "x3"
        value._frame_lock = threading.Lock()
        value.robot_id = "test-robot"
        value._channels = {}
        value._socket = Mock(send=AsyncMock())
        value._close_peer = AsyncMock()
        value._acquire_camera = Mock()
        value._wait_ice_complete = AsyncMock()
        track = VideoStreamTrack()
        try:
            offer = await browser.createOffer()
            offered = SessionDescription.parse(offer.sdp).media[0].rtp.codecs
            assert offered[0].mimeType == "video/VP8"
            with patch("nodes.remote_webrtc_gateway_node.HardwareCameraVideoTrack.create", return_value=track):
                await value._accept_offer(offer.sdp, sender="test-browser")
            response = json.loads(value._socket.send.call_args.args[0])
            codecs = SessionDescription.parse(response["signal"]["sdp"]).media[0].rtp.codecs
            assert len(codecs) == 1
            assert codecs[0].mimeType == "video/H264"
            assert codecs[0].parameters["profile-level-id"] == "42001f"
        finally:
            await browser.close()
            if value._peer is not None:
                await value._peer.close()
            track.stop()
            if getattr(value, "_audio_track", None) is not None:
                value._audio_track.stop()
    asyncio.run(scenario())


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
    assert frame.format.name == "yuv420p"


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


def test_large_jpeg_uses_native_downsampling_before_load_and_preserves_orientation():
    image = Image.new("RGB", (1920, 1080), "red")
    image.paste("blue", (0, 540, 1920, 1080))
    output = BytesIO()
    image.save(output, format="JPEG")
    observations = []
    original = JpegImagePlugin.JpegImageFile.draft

    def draft(source, mode, size):
        assert source.tile  # No full-resolution decode has taken place.
        result = original(source, mode, size)
        observations.append(source.size)
        return result

    with patch.object(JpegImagePlugin.JpegImageFile, "draft", draft):
        frame = CameraVideoTrack._make_frame(output.getvalue(), 320, 240)
    assert observations == [(480, 270)]
    assert (frame.width, frame.height) == (320, 240)
    pixels = frame.to_ndarray(format="rgb24")
    assert pixels[:60, :, 0].mean() > 240
    assert pixels[-60:, :, 2].mean() > 240


@pytest.mark.parametrize("size", [(320, 240), (480, 360)])
@pytest.mark.parametrize("source_size", [(640, 480), (1920, 1080), (160, 120)])
def test_video_resize_and_encoder_colorspace_are_prepared_together(size, source_size):
    image = Image.new("RGB", source_size, "red")
    w, h = source_size
    image.paste("lime", (w // 2, 0, w, h // 2))
    image.paste("blue", (0, h // 2, w // 2, h))
    image.paste("white", (w // 2, h // 2, w, h))
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    # The resize must not return to Pillow, even when draft() cannot reduce
    # the input or the source needs upscaling.
    with patch.object(Image.Image, "resize", side_effect=AssertionError("RGB resize")):
        frame = CameraVideoTrack._make_frame(output.getvalue(), *size)
    assert (frame.width, frame.height) == size
    assert frame.format.name == "yuv420p"
    # Encoder receives this frame directly: another conversion is a no-op.
    assert frame.reformat(format="yuv420p") is frame
    pixels = frame.to_ndarray(format="rgb24")
    width, height = size
    quadrants = [pixels[height // 8:height // 4, width // 8:width // 4],
                 pixels[height // 8:height // 4, width * 5 // 8:width * 3 // 4],
                 pixels[height * 5 // 8:height * 3 // 4, width // 8:width // 4],
                 pixels[height * 5 // 8:height * 3 // 4, width * 5 // 8:width * 3 // 4]]
    assert quadrants[0][:, :, 0].mean() > 240  # Top-left remains red.
    assert quadrants[1][:, :, 1].mean() > 240  # Top-right remains green.
    assert quadrants[2][:, :, 2].mean() > 240  # Bottom-left remains blue.
    assert quadrants[3].mean() > 240  # Bottom-right remains white.


def test_invalid_jpeg_uses_native_placeholder_and_reports_error():
    async def scenario():
        value = Mock(latest_camera_frame=lambda: b"not a jpeg")
        track = CameraVideoTrack(value)
        frame = await track.recv()
        value.report_camera_error.assert_called_once()
        assert (frame.width, frame.height) == (480, 360)
        assert frame.format.name == "yuv420p"
        assert frame.pts is not None
        assert frame.time_base.numerator == 1
        assert frame.time_base.denominator == 90000
        track.stop()
    asyncio.run(scenario())


def test_tcp_retry_selects_a_turn_url_the_sdk_actually_uses():
    servers = [RTCIceServer(urls=[
        "stun:relay.example:3478",
        "turn:relay.example:3478?transport=udp",
        "turn:relay.example:3478?transport=tcp",
    ], username="test-user", credential="test-credential")]
    assert connection_kwargs(servers)["turn_transport"] == "udp"
    assert _ice_servers_for_transport(servers) == servers
    selected = _ice_servers_for_transport(servers, "tcp")
    kwargs = connection_kwargs(selected)
    assert kwargs["turn_transport"] == "tcp"
    assert kwargs["turn_username"] == "test-user"
    assert kwargs["turn_password"] == "test-credential"
    assert selected[0].urls == "stun:relay.example:3478"
    assert len(servers[0].urls) == 3  # The default configuration is unchanged.


def test_tcp_retry_accepts_turns_and_skips_malformed_or_unsupported_urls():
    selected = _ice_servers_for_transport([
        RTCIceServer(urls="not-a-turn-url"),
        RTCIceServer(urls="turn:oauth.example?transport=tcp", credentialType="oauth"),
        RTCIceServer(urls="turns:secure.example:5349", username="test-user", credential="test-credential"),
    ], "tcp")
    kwargs = connection_kwargs(selected)
    assert kwargs["turn_transport"] == "tcp"
    assert kwargs["turn_ssl"] is True
    assert len(selected) == 1


@pytest.mark.parametrize("servers,transport", [
    ([], "tcp"),
    ([RTCIceServer(urls="turn:relay.example?transport=udp")], "tcp"),
    ([RTCIceServer(urls="turns:relay.example?transport=udp")], "tcp"),
    ([], "udp"),
])
def test_tcp_retry_without_supported_config_fails_explicitly(servers, transport):
    with pytest.raises(ValueError):
        _ice_servers_for_transport(servers, transport)


@pytest.mark.parametrize("fields", [
    {"negotiationId": True}, {"negotiationId": -1}, {"negotiationId": 2**53},
    {"negotiationId": "2"}, {"negotiationId": 1.5},
    {"turnTransport": "udp"}, {"turnTransport": {"url": "turn:untrusted.example"}},
])
def test_invalid_offer_metadata_does_not_replace_the_active_peer(fields):
    value = gateway()
    value._accept_offer = AsyncMock()
    asyncio.run(value._handle_signal({"type": "offer", "sdp": "test", **fields}, sender="controller"))
    value._accept_offer.assert_not_called()


def test_tcp_configuration_error_is_reported_without_tearing_down_current_session():
    value = gateway()
    value._peer = Mock()
    value._controller_peer_id = "controller"
    value._ice_servers = []
    value._socket = Mock(send=AsyncMock())
    value._close_peer = AsyncMock()
    asyncio.run(value._accept_offer("test", sender="controller", negotiation_id=2, turn_transport="tcp"))
    value._close_peer.assert_not_called()
    message = json.loads(value._socket.send.call_args.args[0])
    assert message["signal"]["type"] == "error"
    assert message["signal"]["negotiationId"] == 2


def test_stale_candidates_are_ignored_before_parsing():
    value = gateway()
    value._peer = Mock(addIceCandidate=AsyncMock())
    value._negotiation_id = 2
    with patch("nodes.remote_webrtc_gateway_node.candidate_from_sdp") as parse:
        asyncio.run(value._accept_candidate({"negotiationId": 1, "candidate": "stale"}))
        parse.assert_not_called()
    value._peer.addIceCandidate.assert_not_called()


def test_late_peer_teardown_does_not_reset_the_new_session():
    async def scenario():
        value = gateway()
        old_peer = Mock(close=AsyncMock())
        new_peer = Mock(close=AsyncMock())
        value._peer = old_peer
        value._audio_tasks = set()
        value._channels = {"control": Mock()}
        value._servo_gate = Mock()
        closing = asyncio.Event()
        release = asyncio.Event()

        async def close_old():
            closing.set()
            await release.wait()

        old_peer.close.side_effect = close_old
        task = asyncio.create_task(value._close_peer(old_peer))
        await closing.wait()
        value._peer = new_peer
        value._controller_peer_id = "new-controller"
        value._negotiation_id = 2
        value._channels["control"] = new_channel = Mock()
        await value._close_peer(old_peer)  # Late callback from the old peer.
        release.set()
        await task
        assert value._peer is new_peer
        assert value._controller_peer_id == "new-controller"
        assert value._negotiation_id == 2
        assert value._channels["control"] is new_channel
        value._stop_robot_outputs.assert_called_once()
        new_peer.close.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.skipif(os.getenv("WALLE_TEST_TURN_TCP") != "1", reason="Opt-in network probe uses deployment TURN credentials")
def test_deployed_turn_tcp_allocation_and_bidirectional_relay():
    """No ROS publishers, controller seat, microphone or actuator is touched.

    Disable host address gathering and perform ICE in both directions so both
    TURN permissions exist before data is sent. This is not a mobile benchmark.
    """
    import aioice
    stage = "configuration"

    async def scenario():
        nonlocal stage
        kwargs = connection_kwargs(_ice_servers_for_transport(_ice_servers(_remote_config()), "tcp"))
        kwargs.pop("stun_server", None)
        peers = [aioice.Connection(ice_controlling=role, use_ipv4=False, use_ipv6=False,
                                  transport_policy=aioice.TransportPolicy.RELAY, **kwargs)
                 for role in (True, False)]
        completed = False
        try:
            stage = "allocation"
            await asyncio.wait_for(asyncio.gather(*(peer.gather_candidates() for peer in peers)), 10)
            print(json.dumps({"turn_transport": "tcp", "relay_candidates": [
                sum(candidate.type == "relay" for candidate in peer.local_candidates) for peer in peers
            ]}))
            relays = [next(candidate for candidate in peer.local_candidates if candidate.type == "relay")
                      for peer in peers]
            for index, peer in enumerate(peers):
                peer.remote_username = peers[1 - index].local_username
                peer.remote_password = peers[1 - index].local_password
                await peer.add_remote_candidate(relays[1 - index])
                await peer.add_remote_candidate(None)
                assert all(candidate.type == "relay" for candidate in peer.local_candidates)
            stage = "ice-handshake"
            await asyncio.wait_for(asyncio.gather(*(peer.connect() for peer in peers)), 15)
            samples = []
            for seq in range(20):
                payload = f"turn-tcp-probe-{seq}".encode()
                start = time.monotonic()
                stage = f"forward-{seq}"
                await peers[0].send(payload)
                assert await asyncio.wait_for(peers[1].recv(), 2) == payload
                stage = f"return-{seq}"
                await peers[1].send(payload)
                assert await asyncio.wait_for(peers[0].recv(), 2) == payload
                samples.append((time.monotonic() - start) * 1000)
            completed = True
            print(json.dumps({"turn_transport": "tcp", "relay_roundtrips": len(samples),
                              "rtt_mean_ms": round(sum(samples) / len(samples), 1),
                              "rtt_max_ms": round(max(samples), 1)}))
        finally:
            try:
                await asyncio.wait_for(asyncio.gather(*(peer.close() for peer in peers)), 5)
            except asyncio.TimeoutError:
                print("TURN TCP probe cleanup timed out")
                if completed:
                    stage = "cleanup"
                    raise
    try:
        asyncio.run(asyncio.wait_for(scenario(), 55))
    except Exception as exc:
        # Avoid credential-bearing tracebacks from network client internals.
        pytest.fail(f"TURN TCP probe failed at {stage} ({type(exc).__name__})", pytrace=False)
