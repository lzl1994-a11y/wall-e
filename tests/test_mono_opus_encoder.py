from fractions import Fraction

import numpy as np
import pytest
from av import AudioFrame
from aiortc.codecs.opus import OpusDecoder
from aiortc.jitterbuffer import JitterFrame

from services.remote.mono_opus_encoder import MonoOpusEncoder


def frame(pts, *, samples=960, layout="mono", rate=48000):
    result = AudioFrame(format="s16", layout=layout, samples=samples)
    signal = (np.sin((np.arange(samples) + pts) * 2 * np.pi * 440 / 48000) * 6000).astype("int16")
    data = signal.tobytes() * (2 if layout == "stereo" else 1)
    result.planes[0].update(data)
    result.sample_rate = rate
    result.time_base = Fraction(1, rate)
    result.pts = pts
    return result


def test_real_mono_opus_is_playable_by_stock_webrtc_decoder_with_continuous_clock():
    encoder = MonoOpusEncoder()
    decoder = OpusDecoder()
    decoded = []
    timestamps = []
    for pts in range(9600, 9600 + 960 * 20, 960):
        payloads, timestamp = encoder.encode(frame(pts))
        assert len(payloads) == 1
        # Opus TOC's stereo bit reflects actual bitstream channel count;
        # WebRTC's two-channel SDP codec can carry mono speech.
        assert not payloads[0][0] & 4
        timestamps.append(timestamp)
        decoded.extend(decoder.decode(JitterFrame(payloads[0], timestamp)))
    assert timestamps == list(range(0, 960 * 20, 960))
    assert sum(value.samples for value in decoded) == 960 * 20
    signal = np.concatenate([value.to_ndarray().reshape(-1, 2)[:, 0] for value in decoded])
    peak = np.fft.rfftfreq(len(signal), 1 / 48000)[np.argmax(abs(np.fft.rfft(signal)))]
    assert abs(peak - 440) < 5
    assert np.sqrt(np.mean(signal.astype(float) ** 2)) > 1000


@pytest.mark.parametrize("kwargs", [{"samples": 480}, {"layout": "stereo"}, {"rate": 16000}])
def test_unexpected_source_frame_is_rejected_instead_of_silently_reframed(kwargs):
    with pytest.raises(ValueError, match="20 ms"):
        MonoOpusEncoder().encode(frame(0, **kwargs))
