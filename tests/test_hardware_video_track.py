import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
pytest.importorskip("aiortc")
pytest.importorskip("av")
from aiortc.mediastreams import MediaStreamError
from services.remote.hardware_video_track import HardwareCameraVideoTrack, hardware_h264_offered


def offer(profile='42001f',mode='1'):
    return ('v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
            'm=video 9 UDP/TLS/RTP/SAVPF 99\r\nc=IN IP4 0.0.0.0\r\n'
            'a=rtpmap:99 H264/90000\r\n'
            f'a=fmtp:99 profile-level-id={profile};packetization-mode={mode}\r\n')


def test_hardware_requires_matching_profile_and_packetization():
    assert hardware_h264_offered(offer())
    assert not hardware_h264_offered(offer('64001f'))
    assert not hardware_h264_offered(offer('42e01f'))
    assert not hardware_h264_offered(offer(mode='0'))


def test_packet_pts_tracks_elapsed_time_and_profile():
    async def run():
        gateway=Mock(latest_raw_camera_frame=lambda:(b'jpeg',time.monotonic()))
        codec=Mock(encode=lambda *args:b'\x00\x00\x01\x65test')
        track=HardwareCameraVideoTrack(gateway,Path('unused'))
        with patch('services.remote.hardware_video_track.X3VideoCodec',return_value=codec):
            first=await track.recv()
            track._started_at-=2;track.profile='low'
            second=await track.recv()
            assert second.pts-first.pts>=180000
            assert second.time_base.denominator==90000
            await track.aclose()
        codec.close.assert_called_once()
    asyncio.run(run())


def test_cancelled_encode_is_finished_before_sdk_memory_is_released():
    async def run():
        started=threading.Event();finish=threading.Event();order=[]
        def encode(*args):
            started.set();finish.wait(2);order.append('encode');return b'packet'
        codec=Mock(encode=encode,close=lambda:order.append('close'))
        gateway=Mock(latest_raw_camera_frame=lambda:(b'jpeg',time.monotonic()))
        track=HardwareCameraVideoTrack(gateway,Path('unused'))
        with patch('services.remote.hardware_video_track.X3VideoCodec',return_value=codec):
            task=asyncio.create_task(track.recv())
            assert await asyncio.to_thread(started.wait,1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
            closing=asyncio.create_task(track.aclose())
            await asyncio.sleep(.03);assert not order
            finish.set();await closing
            assert order==['encode','close']
            with pytest.raises(MediaStreamError):await track.recv()
    asyncio.run(run())


def test_stalled_camera_fails_explicitly_without_encoding_old_frame():
    async def run():
        gateway=Mock(latest_raw_camera_frame=lambda:(b'jpeg',time.monotonic()-2))
        track=HardwareCameraVideoTrack(gateway,Path('unused'))
        awaitable=track.recv()
        with pytest.raises(MediaStreamError):await awaitable
        gateway.report_hardware_video_error.assert_called_once()
        assert track._codec is None
        await track.aclose()
    asyncio.run(run())
