"""Bounded Opus reordering without requiring consecutive received packets.

RFC 7587 section 4.2 makes each RTP payload one complete Opus packet. A
missing packet therefore must not hold later complete packets indefinitely.
aiortc 1.15.0 instead requires four consecutive frames, which discards good
audio under periodic loss. Keep its four-packet reorder window, but count
sequence positions (including losses) rather than consecutive good frames.
"""
from __future__ import annotations

import aiortc
from aiortc.jitterbuffer import JitterBuffer, JitterFrame
from aiortc.sdp import SessionDescription


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
        setattr(receiver, field, OpusAudioJitterBuffer())
        installed += 1
    return installed
