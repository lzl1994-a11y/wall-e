"""Bounded, local speaker reference; never waits for the microphone worker."""
from __future__ import annotations

import errno
import os
import socket
import struct
import time

import numpy as np

RATE = 48000
FRAME_BYTES = 1920
HEADER = struct.Struct("<4sdH")
MAGIC = b"AEC1"
MAX_AGE_SEC = 0.25


def address():
    # Abstract Linux sockets disappear automatically after a crash/restart.
    return f"\0walle.echo-reference.{os.getuid()}"


class EchoReferenceSender:
    def __init__(self):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self.dropped = 0
        self.device_gain = 1

    def configure_device(self, name):
        # Ear S3 firmware i2s_mic_driver_write_tx: saturating int16 * 4.
        # Other USB speakers retain the ordinary unity reference.
        self.device_gain = 4 if str(name).startswith("Walle Ear S3:") else 1

    def send(self, audio, latency):
        # Exactly the clipped final float32 mixture passed to the sound card.
        pcm = np.rint(np.clip(audio * self.device_gain, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
        if len(pcm) != FRAME_BYTES:
            raise ValueError("echo reference must be 48 kHz mono / 20 ms")
        self._send(pcm, latency)

    def end(self):
        self._send(b"", 0)

    def _send(self, pcm, latency):
        message = HEADER.pack(MAGIC, time.monotonic(), min(500, max(0, round(latency * 1000)))) + pcm
        try:
            self._socket.sendto(message, address())
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED, errno.EAGAIN, errno.ENOBUFS):
                raise
            self.dropped += 1
            if self.dropped == 1:
                print("[Playback Service] AEC 参考接收端不可用或拥塞", flush=True)

    def close(self):
        self._socket.close()


class EchoReferenceReceiver:
    def __init__(self):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, FRAME_BYTES * 8)
            self._socket.bind(address())
            self._socket.setblocking(False)
        except Exception:
            self._socket.close()
            raise
        self.dropped = 0
        self.frames = 0

    def drain(self):
        messages = []
        # Bound work even if a producer misbehaves. 16 blocks = 320 ms.
        for _ in range(16):
            try:
                data, ancillary, flags, _ = self._socket.recvmsg(HEADER.size + FRAME_BYTES,
                                                              socket.CMSG_SPACE(12))
            except BlockingIOError:
                break
            credentials = [struct.unpack("3i", value) for level, kind, value in ancillary
                           if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS]
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not credentials or credentials[0][1] != os.getuid():
                self.dropped += 1
                continue
            packet = decode_reference(data, time.monotonic())
            if packet is None:
                self.dropped += 1
                continue
            messages.append(packet)
            if packet[0]:
                self.frames += 1
        return messages

    def close(self):
        self._socket.close()


def decode_reference(data, now):
    if len(data) not in (HEADER.size, HEADER.size + FRAME_BYTES):
        return None
    magic, sent_at, delay_ms = HEADER.unpack_from(data)
    age = now - sent_at
    if magic != MAGIC or not 0 <= age <= MAX_AGE_SEC or delay_ms > 500:
        return None
    return data[HEADER.size:], delay_ms, sent_at
