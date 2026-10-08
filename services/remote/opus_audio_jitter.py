"""Bounded Opus reordering without requiring consecutive received packets.

RFC 7587 section 4.2 makes each RTP payload one complete Opus packet. A
missing packet therefore must not hold later complete packets indefinitely.
aiortc 1.15.0 instead requires four consecutive frames, which discards good
audio under periodic loss. Keep its four-packet reorder window, but count
sequence positions (including losses) rather than consecutive good frames.
"""
from __future__ import annotations

import aiortc
import queue
from aiortc.codecs import get_decoder
from aiortc.jitterbuffer import JitterBuffer, JitterFrame
from aiortc.sdp import SessionDescription


class AudioDecoderQueue(queue.Queue):
    """Decode small audio blocks on the receiver loop without a thread hop.

    The SDK worker waits on this queue only for its normal shutdown sentinel.
    Video retains its worker. Audio frames go directly to the existing track
    queue on the same event loop that receives their encoded RTP packets.
    """
    def __init__(self, output):
        super().__init__()
        self._output = output
        self._decoder = None
        self._codec_name = None

    def put(self, task, block=True, timeout=None):
        if task is None:
            self._decoder = None
            return super().put(task, block=block, timeout=timeout)
        codec, encoded = task
        if codec.name != self._codec_name:
            self._decoder = get_decoder(codec)
            self._codec_name = codec.name
        for frame in self._decoder.decode(encoded):
            self._output.put_nowait(frame)


class OpusAudioJitterBuffer:
    capacity = 16
    reorder_packets = 4

    def __init__(self):
        self._origin = None
        self._latest = None
        self._packets = {}
        self.missing_packets = 0
        self.late_packets = 0
        self.duplicate_packets = 0
        self.overflow_packets = 0

    def add(self, packet):
        sequence = packet.sequence_number
        if self._latest is None:
            self._origin = self._latest = sequence
        else:
            delta = (sequence - self._latest) & 0xffff
            sequence = self._latest + (delta if delta < 0x8000 else delta - 0x10000)
        if sequence < self._origin:
            self.late_packets += 1
            return False, None
        if sequence in self._packets:
            self.duplicate_packets += 1
            return False, None
        self._latest = max(self._latest, sequence)

        # A discontinuity cannot allocate an unbounded queue or scan a huge gap.
        first = self._latest - self.capacity + 1
        if self._origin < first:
            expired = [key for key in self._packets if key < first]
            self.overflow_packets += len(expired)
            self.missing_packets += first - self._origin - len(expired)
            for key in expired:
                del self._packets[key]
            self._origin = first
        self._packets[sequence] = packet

        # A packet can arrive out of order until four later sequence positions
        # have arrived. After that deadline, skip only the missing position.
        deadline = self._latest - self.reorder_packets
        while self._origin <= deadline:
            current = self._packets.pop(self._origin, None)
            self._origin += 1
            if current is not None:
                return False, JitterFrame(current._data, current.timestamp)
            self.missing_packets += 1
        return False, None


def install_opus_audio_jitter(peer, answer_sdp: str) -> int:
    """Install before setLocalDescription starts the receiver transports.

    aiortc has no public jitter-buffer hook. Isolate the one private attribute
    here and require the tested/pinned release and original buffer shape;
    never silently run an unverified SDK integration. Video and other codecs
    retain their normal receivers. No installed dependency files are edited.
    """
    audio = [media for media in SessionDescription.parse(answer_sdp).media
             if media.kind == "audio"]
    if not audio or not audio[0].rtp.codecs or audio[0].rtp.codecs[0].mimeType.lower() != "audio/opus":
        return 0
    if aiortc.__version__ != "1.15.0":
        raise RuntimeError("Opus audio receiver requires tested aiortc==1.15.0")
    installed = 0
    for transceiver in peer.getTransceivers():
        if transceiver.kind != "audio":
            continue
        receiver = transceiver.receiver
        field = "_RTCRtpReceiver__jitter_buffer"
        previous = getattr(receiver, field, None)
        if isinstance(previous, OpusAudioJitterBuffer):
            continue
        if type(previous) is not JitterBuffer or previous.capacity != 16 or previous._origin is not None:
            raise RuntimeError("Unexpected aiortc audio jitter buffer state")
        decoder_queue = getattr(receiver, "_RTCRtpReceiver__decoder_queue", None)
        if (type(decoder_queue) is not queue.Queue or not decoder_queue.empty()
                or getattr(receiver, "_RTCRtpReceiver__decoder_thread", None) is not None
                or (receiver.track is not None and not hasattr(receiver.track, "_queue"))):
            raise RuntimeError("Unexpected aiortc audio decoder state")
        # A browser receiving only robot audio has no incoming audio track.
        if receiver.track is not None:
            receiver._RTCRtpReceiver__decoder_queue = AudioDecoderQueue(receiver.track._queue)
        setattr(receiver, field, OpusAudioJitterBuffer())
        installed += 1
    return installed
