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


def test_normal_mode_follows_fresh_source_without_ten_fps_wait():
    async def run():
        clock = [100.0]
        source = [b"first", 100.0]
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay
            source[:] = [b"second", clock[0]]

        gateway = Mock(latest_raw_camera_frame=lambda: tuple(source))
        codec = Mock(encode=Mock(return_value=b"packet"))
        with patch("services.remote.hardware_video_track.time.monotonic", side_effect=lambda: clock[0]), \
                patch("services.remote.hardware_video_track.asyncio.sleep", side_effect=sleep), \
                patch("services.remote.hardware_video_track.X3VideoCodec", return_value=codec):
            track = HardwareCameraVideoTrack(gateway, Path("unused"))
            try:
                first = await track.recv()
                assert not any(sleeps)
                # A fresh camera frame arrives 1/15 second later. The old
                # normal-mode timer would still sleep until 1/10 second.
                clock[0] += 1 / 15
                source[:] = [b"next", clock[0]]
                second = await track.recv()
                assert not any(sleeps)
                assert second.pts > first.pts
                # Asking again with unchanged source must wait for new input.
                third = await track.recv()
                assert sleeps == [.002]
                assert third.pts > second.pts
                assert [call.args[0] for call in codec.encode.call_args_list] == [b"first", b"next", b"second"]
                assert all(call.args[1:4] == (480, 360, 10) for call in codec.encode.call_args_list)
            finally:
                await track.aclose()
    asyncio.run(run())


def test_stop_ends_wait_for_fresh_source_without_another_encode():
    async def run():
        captured_at = time.monotonic()
        gateway = Mock(latest_raw_camera_frame=lambda: (b"jpeg", captured_at))
        codec = Mock(encode=Mock(return_value=b"packet"))
        with patch("services.remote.hardware_video_track.X3VideoCodec", return_value=codec):
            track = HardwareCameraVideoTrack(gateway, Path("unused"))
            try:
                await track.recv()
                waiting = asyncio.create_task(track.recv())
                await asyncio.sleep(.01)
                assert not waiting.done()
                track.stop()
                with pytest.raises(MediaStreamError):
                    await asyncio.wait_for(waiting, .2)
                assert codec.encode.call_count == 1
            finally:
                await track.aclose()
    asyncio.run(run())


def test_uncapped_frames_do_not_accumulate_future_wait_when_switching_to_low():
    async def run():
        clock = [100.0]
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        gateway = Mock(latest_raw_camera_frame=lambda: (b"jpeg", clock[0]))
        codec = Mock(encode=Mock(return_value=b"packet"))
        with patch("services.remote.hardware_video_track.time.monotonic", side_effect=lambda: clock[0]), \
                patch("services.remote.hardware_video_track.asyncio.sleep", side_effect=sleep), \
                patch("services.remote.hardware_video_track.X3VideoCodec", return_value=codec):
            track = HardwareCameraVideoTrack(gateway, Path("unused"))
            try:
                for _ in range(100):
                    clock[0] += 1 / 15
                    await track.recv()
                assert sleeps == []
                track.profile = "low"
                clock[0] += 1 / 15
                await track.recv()
                assert sleeps == [0]
                await track.recv()
                assert sleeps == pytest.approx([0, 2 / 15])
                await track.recv()
                assert sleeps == pytest.approx([0, 2 / 15, .2])
            finally:
                await track.aclose()
    asyncio.run(run())


def test_low_mode_keeps_five_fps_wait():
    async def run():
        clock = [100.0]
        sleeps = []

        async def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        gateway = Mock(latest_raw_camera_frame=lambda: (b"jpeg", clock[0]))
        codec = Mock(encode=Mock(return_value=b"packet"))
        with patch("services.remote.hardware_video_track.time.monotonic", side_effect=lambda: clock[0]), \
                patch("services.remote.hardware_video_track.asyncio.sleep", side_effect=sleep), \
                patch("services.remote.hardware_video_track.X3VideoCodec", return_value=codec):
            track = HardwareCameraVideoTrack(gateway, Path("unused"))
            track.profile = "low"
            try:
                await track.recv()
                await track.recv()
                assert sleeps == pytest.approx([0, .2])
                assert all(call.args[1:4] == (320, 240, 5) for call in codec.encode.call_args_list)
            finally:
                await track.aclose()
    asyncio.run(run())


def test_repeated_source_frame_stalls_explicitly_and_does_not_reencode():
    async def run():
        clock = [100.0]

        async def sleep(delay):
            clock[0] += delay

        gateway = Mock(latest_raw_camera_frame=lambda: (b"jpeg", 100.0))
        codec = Mock(encode=Mock(return_value=b"packet"))
        with patch("services.remote.hardware_video_track.time.monotonic", side_effect=lambda: clock[0]), \
                patch("services.remote.hardware_video_track.asyncio.sleep", side_effect=sleep), \
                patch("services.remote.hardware_video_track.X3VideoCodec", return_value=codec):
            track = HardwareCameraVideoTrack(gateway, Path("unused"))
            try:
                await track.recv()
                with pytest.raises(MediaStreamError):
                    await track.recv()
                assert codec.encode.call_count == 1
                gateway.report_hardware_video_error.assert_called_once()
            finally:
                await track.aclose()
    asyncio.run(run())
