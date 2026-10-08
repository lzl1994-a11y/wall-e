"""Bounded audio bursts must survive an ordinary media-loop scheduling pause."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("aiortc")
from nodes.remote_webrtc_gateway_node import RobotAudioTrack


def push_blocks(track, values):
    for value in values:
        track.push(np.full(960, value, dtype=np.int16).tobytes())


def clock(value):
    # Patch this module's clock reference, not asyncio's global monotonic clock.
    return patch("nodes.remote_webrtc_gateway_node.time", SimpleNamespace(monotonic=lambda: value))


def test_six_buffered_audio_frames_are_not_discarded_during_catchup():
    async def scenario():
        with clock(100):
            track = RobotAudioTrack()
            push_blocks(track, range(1, 7))
        with clock(100.12):
            frames = [await track.recv() for _ in range(6)]
        assert [int(frame.to_ndarray()[0, 0]) for frame in frames] == list(range(1, 7))
        assert [frame.pts for frame in frames] == [n * 960 for n in range(6)]
        track.stop()
    asyncio.run(scenario())


def test_eighty_ms_pause_catches_up_without_creating_timestamp_holes():
    async def scenario():
        with clock(100):
            track = RobotAudioTrack()
            push_blocks(track, range(1, 5))
        with clock(100.08):
            frames = [await track.recv() for _ in range(4)]
        assert [frame.pts for frame in frames] == [0, 960, 1920, 2880]
        track.stop()
    asyncio.run(scenario())


def test_large_source_backlog_is_bounded_and_keeps_newest_audio():
    async def scenario():
        with clock(100):
            track = RobotAudioTrack()
            push_blocks(track, range(1, 31))
        assert len(track._buffer) == 48000 * 2 // 5
        with clock(100.2):
            frames = [await track.recv() for _ in range(10)]
        assert [int(frame.to_ndarray()[0, 0]) for frame in frames] == list(range(21, 31))
        track.stop()
    asyncio.run(scenario())


def test_long_pause_without_source_does_not_send_seconds_of_stale_frames():
    async def scenario():
        with clock(100):
            track = RobotAudioTrack()
        with clock(102):
            frame = await track.recv()
        assert frame.pts == 96000
        assert not frame.to_ndarray().any()
        track.stop()
    asyncio.run(scenario())
