"""Encode the robot's already framed mono speech without stereo resampling."""
from fractions import Fraction

from av import CodecContext


class MonoOpusEncoder:
    def __init__(self):
        self.codec = CodecContext.create("libopus", "w")
        self.codec.bit_rate = 32000
        self.codec.format = "s16"
        self.codec.layout = "mono"
        self.codec.sample_rate = 48000
        self.codec.time_base = Fraction(1, 48000)
        self.codec.options = {"application": "voip", "compression_level": "5"}
        self.first_packet_pts = None

    def encode(self, frame):
        if (frame.format.name != "s16" or frame.layout.name != "mono"
                or frame.sample_rate != 48000 or frame.samples != 960):
            raise ValueError("Expected one 20 ms, 48 kHz, s16 mono speech frame")
        packets = self.codec.encode(frame)
        if not packets:
            return [], None
        if self.first_packet_pts is None:
            self.first_packet_pts = packets[0].pts
        return [bytes(packet) for packet in packets], packets[0].pts - self.first_packet_pts
