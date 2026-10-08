"""Loss must discard missing Opus packets, not the good packets behind them."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("aiortc")
from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtp import RtpPacket
from services.remote.opus_audio_jitter import OpusAudioJitterBuffer, install_opus_audio_jitter


def buffer():
    return OpusAudioJitterBuffer()


def packet(sequence, timestamp=None):
    value = RtpPacket(sequence_number=sequence & 0xffff,
                      timestamp=(sequence * 960 if timestamp is None else timestamp) & 0xffffffff)
    value._data = sequence.to_bytes(4, "big")
    return value


def feed(value, sequences):
    emitted = []
    for sequence in sequences:
        pli, frame = value.add(packet(sequence))
        assert not pli
        if frame is not None:
            emitted.append(int.from_bytes(frame.data, "big"))
    return emitted


def test_periodic_loss_does_not_discard_received_good_frames():
    sequences = [n for n in range(600) if n % 6 != 5]
    emitted = feed(buffer(), sequences)
    assert emitted == [n for n in sequences if n <= sequences[-1] - 4]


def test_one_loss_resumes_after_the_reorder_window():
    emitted = feed(buffer(), [n for n in range(40) if n != 8])
    assert emitted == [n for n in range(36) if n != 8]


def test_reordering_inside_window_preserves_all_packets():
    sequences = list(range(30))
    sequences[8:11] = [10, 8, 9]
    assert feed(buffer(), sequences) == list(range(26))


def test_sequence_wrap_and_rtp_timestamp_are_preserved():
    value = buffer()
    assert feed(value, range(65530, 65550)) == list(range(65530, 65546))
    packet_value = packet(65550, timestamp=17)
    _, frame = value.add(packet_value)
    assert frame.timestamp == (65546 * 960) & 0xffffffff


def test_late_and_duplicate_packets_do_not_play_twice():
    value = OpusAudioJitterBuffer()
    emitted = feed(value, range(20))
    assert feed(value, [3, 19]) == []
    assert emitted == list(range(16))
    assert value.late_packets == 1
    assert value.duplicate_packets == 1


def test_large_gap_has_bounded_storage_and_recovers():
    value = OpusAudioJitterBuffer()
    feed(value, range(20))
    assert feed(value, range(10000, 10020)) == list(range(10000, 10016))
    assert len(value._packets) <= value.capacity
    assert value.overflow_packets == 4


def test_empty_buffer_waits_for_reorder_window():
    value = OpusAudioJitterBuffer()
    assert feed(value, range(4)) == []
    assert feed(value, [4]) == [0]


OPUS_SDP = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n"


def receiver(kind="audio", value=None):
    import asyncio
    import queue
    obj = SimpleNamespace(track=SimpleNamespace(kind=kind, _queue=asyncio.Queue()))
    obj._RTCRtpReceiver__decoder_queue = queue.Queue()
    obj._RTCRtpReceiver__decoder_thread = None
    setattr(obj, "_RTCRtpReceiver__jitter_buffer", value if value is not None else JitterBuffer(16, 4))
    return obj


def peer_for(*receivers):
    return SimpleNamespace(getTransceivers=lambda: [
        SimpleNamespace(kind=value.track.kind, receiver=value) for value in receivers])


def test_adapter_changes_only_opus_audio_and_is_idempotent():
    audio, video = receiver(), receiver("video")
    original = video._RTCRtpReceiver__jitter_buffer
    peer = peer_for(audio, video)
    assert install_opus_audio_jitter(peer, OPUS_SDP) == 1
    assert isinstance(audio._RTCRtpReceiver__jitter_buffer, OpusAudioJitterBuffer)
    assert video._RTCRtpReceiver__jitter_buffer is original
    assert install_opus_audio_jitter(peer, OPUS_SDP) == 0


def test_other_codec_keeps_original_receiver():
    audio = receiver()
    original = audio._RTCRtpReceiver__jitter_buffer
    peer = peer_for(audio)
    assert install_opus_audio_jitter(peer, OPUS_SDP.replace("opus/48000/2", "PCMU/8000")) == 0
    assert audio._RTCRtpReceiver__jitter_buffer is original


def test_unknown_sdk_version_fails_explicitly():
    with patch("services.remote.opus_audio_jitter.aiortc.__version__", "2.0.0"):
        with pytest.raises(RuntimeError, match="aiortc==1.15.0"):
            install_opus_audio_jitter(peer_for(), OPUS_SDP)


def test_already_running_receiver_is_rejected():
    value = JitterBuffer(16, 4)
    value.add(packet(0))
    peer = peer_for(receiver(value=value))
    with pytest.raises(RuntimeError, match="Unexpected"):
        install_opus_audio_jitter(peer, OPUS_SDP)


def test_started_decoder_is_rejected_before_replacing_the_jitter_buffer():
    audio = receiver()
    audio._RTCRtpReceiver__decoder_thread = object()
    previous = audio._RTCRtpReceiver__jitter_buffer
    with pytest.raises(RuntimeError, match="decoder state"):
        install_opus_audio_jitter(peer_for(audio), OPUS_SDP)
    assert audio._RTCRtpReceiver__jitter_buffer is previous
