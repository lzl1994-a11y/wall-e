import asyncio
from fractions import Fraction

import numpy as np
import pytest
from av import AudioFrame
from aiortc.codecs.opus import OpusEncoder
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtcrtpparameters import RTCRtpCodecParameters


def test_audio_decode_delivers_real_pcm_immediately_without_a_thread_round_trip():
    from services.remote.opus_audio_jitter import AudioDecoderQueue
    output = asyncio.Queue()
    queue = AudioDecoderQueue(output)
    encoder = OpusEncoder()
    codec = RTCRtpCodecParameters(mimeType="audio/opus", clockRate=48000, channels=2)
    frames = []
    for index in range(20):
        frame = AudioFrame(format="s16", layout="mono", samples=960)
        samples = (np.sin((np.arange(960) + index * 960) * 2 * np.pi * 440 / 48000) * 6000).astype("int16")
        frame.planes[0].update(samples.tobytes())
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frame.pts = index * 960
        payloads, timestamp = encoder.encode(frame)
        queue.put((codec, JitterFrame(payloads[0], timestamp)))
        assert queue.empty()
        assert output.qsize() == 1
        frames.append(output.get_nowait())
    assert all(frame.layout.name == "mono" and frame.samples == 960 for frame in frames)
    assert [frame.pts for frame in frames] == list(range(0, 960 * 20, 960))
    samples = np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
    peak = np.fft.rfftfreq(len(samples), 1 / 48000)[np.argmax(abs(np.fft.rfft(samples)))]
    assert abs(peak - 440) < 5
    assert np.sqrt(np.mean(samples.astype(float) ** 2)) > 1000
    queue.put(None)
    assert queue.get_nowait() is None  # Existing SDK shutdown worker is unblocked.


def test_decode_queue_cannot_build_unbounded_playback_delay():
    from services.remote.opus_audio_jitter import AudioDecoderQueue
    output = asyncio.Queue()
    for _ in range(16):
        output.put_nowait(object())
    with pytest.raises(RuntimeError, match="bounded"):
        AudioDecoderQueue(output).put((object(), object()))
    assert output.qsize() == 16


def test_buffered_datagrams_do_not_starve_playback_and_audio_sender():
    from types import SimpleNamespace
    from services.remote.opus_audio_jitter import install_audio_transport_fairness

    async def scenario():
        output = asyncio.Queue()
        received = []
        played = []
        sent = []

        async def read_one():
            index = len(received)
            received.append(index)
            output.put_nowait(index)
            if output.qsize() > 16:
                raise RuntimeError("PCM burst starved its consumer")

        transport = SimpleNamespace(_recv_next=read_one)
        install_audio_transport_fairness(transport)
        installed = transport._recv_next
        install_audio_transport_fairness(transport)
        assert transport._recv_next is installed

        async def receive():
            for _ in range(100):
                await transport._recv_next()

        async def play_and_send():
            for _ in range(100):
                played.append(await output.get())
                sent.append(len(received))

        await asyncio.gather(receive(), play_and_send())
        assert played == list(range(100))
        assert sent[0] < 16  # Sending runs during a burst, before it finishes.

    asyncio.run(scenario())
