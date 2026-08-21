import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from services.camera_frame import save_camera_photo
from services.tft_preview_server import (
    EXPECTED_DEVICE_ID,
    HEADER,
    HELLO,
    JPEG_FRAME,
    MAGIC,
    PING,
    PONG,
    PROTOCOL_VERSION,
    STREAM_END,
    STREAM_START,
    STREAM_START_MESSAGE,
    TftPreviewServer,
    TftPreviewSettings,
    decode_header,
    encode_message,
)


def _source_jpeg(width=640, height=360):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (0, 0, 255)
    image[:, width // 2:] = (0, 255, 0)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise AssertionError("failed to build fake JPEG")
    return encoded.tobytes()


class _FakeFrameProvider:
    def __init__(self, frame, count=4):
        self.frame = frame
        self.count = count
        self.calls = []

    def capture_stream(self, **kwargs):
        self.calls.append(kwargs)
        for _ in range(self.count):
            kwargs["on_frame"](self.frame)
        return self.frame


class _FakeClock:
    def __init__(self):
        self.values = iter((10.0, 13.0, 20.0, 23.0, 30.0, 33.0))

    def __call__(self):
        return next(self.values)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))

    def error(self, message):
        self.messages.append(("error", message))


def _recv_exact(client, size):
    output = bytearray()
    while len(output) < size:
        chunk = client.recv(size - len(output))
        if not chunk:
            raise ConnectionError("test client disconnected")
        output.extend(chunk)
    return bytes(output)


def _recv_message(client):
    header = _recv_exact(client, HEADER.size)
    message_type, flags, sequence, length = decode_header(header)
    payload = _recv_exact(client, length) if length else b""
    return message_type, flags, sequence, payload


def _wait_until(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _sof_marker(jpeg):
    index = 2
    while index + 3 < len(jpeg):
        if jpeg[index] != 0xFF:
            index += 1
            continue
        marker = jpeg[index + 1]
        index += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            return None
        segment_length = int.from_bytes(jpeg[index:index + 2], "big")
        if marker in {0xC0, 0xC1, 0xC2}:
            return marker
        index += segment_length
    return None


class TftProtocolTests(unittest.TestCase):
    def test_header_is_exactly_16_bytes_and_network_order(self):
        packet = encode_message(JPEG_FRAME, 0x01020304, b"jpeg")

        self.assertEqual(HEADER.size, 16)
        self.assertEqual(packet[:4], MAGIC)
        self.assertEqual(packet[4], PROTOCOL_VERSION)
        self.assertEqual(packet[5], JPEG_FRAME)
        self.assertEqual(packet[6:8], b"\x00\x00")
        self.assertEqual(packet[8:12], b"\x01\x02\x03\x04")
        self.assertEqual(packet[12:16], b"\x00\x00\x00\x04")
        self.assertEqual(decode_header(packet[:16]), (JPEG_FRAME, 0, 0x01020304, 4))


class TftPreviewServerTests(unittest.TestCase):
    def setUp(self):
        self.logger = _Logger()
        self.server = TftPreviewServer(
            TftPreviewSettings(bind_address="127.0.0.1", port=0),
            logger=self.logger,
            clock=_FakeClock(),
        )
        self.server.start()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            try:
                client.close()
            except OSError:
                pass
        self.server.stop()

    def _connect(self, *, fragmented=False, sticky_ping_sequence=None):
        client = socket.create_connection(("127.0.0.1", self.server.port), timeout=1.0)
        client.settimeout(1.0)
        self.clients.append(client)
        hello = encode_message(HELLO, 7, EXPECTED_DEVICE_ID.encode("ascii"))
        if fragmented:
            client.sendall(hello[:3])
            client.sendall(hello[3:11])
            tail = hello[11:]
            if sticky_ping_sequence is not None:
                tail += encode_message(PING, sticky_ping_sequence)
            client.sendall(tail)
        else:
            client.sendall(hello)
        self.assertTrue(_wait_until(lambda: self.server.device_connected))
        return client

    def test_fragmented_hello_and_sticky_ping_are_parsed_and_pong_keeps_sequence(self):
        client = self._connect(fragmented=True, sticky_ping_sequence=0x12345678)

        message_type, flags, sequence, payload = _recv_message(client)

        self.assertEqual((message_type, flags, sequence, payload), (PONG, 0, 0x12345678, b""))
        self.assertTrue(any("device_id=WALL_E_TFT" in text for _, text in self.logger.messages))

    def test_preview_order_parameters_sequences_and_jpeg_requirements(self):
        client = self._connect()
        provider = _FakeFrameProvider(_source_jpeg(), count=4)

        result = self.server.send_camera_preview(
            provider,
            duration_ms=3000,
            hold_ms=3000,
            fps=10,
        )
        messages = [_recv_message(client) for _ in range(6)]

        self.assertEqual([item[0] for item in messages], [
            STREAM_START_MESSAGE,
            JPEG_FRAME,
            JPEG_FRAME,
            JPEG_FRAME,
            JPEG_FRAME,
            STREAM_END,
        ])
        self.assertEqual(STREAM_START.unpack(messages[0][3]), (3000, 3000, 10, 0))
        self.assertEqual(messages[1][2], (messages[0][2] << 16) | 0)
        self.assertEqual(messages[4][2], (messages[0][2] << 16) | 3)
        self.assertEqual(result.sent_frames, 4)
        self.assertEqual(result.last_frame, provider.frame)
        self.assertEqual(provider.calls[0]["duration_ms"], 3000)

        for message in messages[1:5]:
            jpeg = message[3]
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))
            self.assertTrue(jpeg.endswith(b"\xff\xd9"))
            self.assertLessEqual(len(jpeg), 256 * 1024)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape[:2], (240, 240))
            self.assertEqual(_sof_marker(jpeg), 0xC0, "JPEG must use baseline SOF0")

    def test_disconnect_then_reconnect_can_send_another_preview(self):
        first = self._connect()
        first.close()
        self.assertTrue(_wait_until(lambda: not self.server.device_connected))

        second = self._connect()
        result = self.server.send_camera_preview(
            _FakeFrameProvider(_source_jpeg(), count=1),
            duration_ms=1500,
            hold_ms=3000,
            fps=10,
        )
        messages = [_recv_message(second) for _ in range(3)]

        self.assertEqual([item[0] for item in messages], [
            STREAM_START_MESSAGE,
            JPEG_FRAME,
            STREAM_END,
        ])
        self.assertEqual(STREAM_START.unpack(messages[0][3]), (1500, 3000, 10, 0))
        self.assertEqual(result.sent_frames, 1)

    def test_no_network_still_captures_and_saves_original_photo(self):
        frame = _source_jpeg()
        provider = _FakeFrameProvider(frame, count=3)

        result = self.server.send_camera_preview(provider, duration_ms=3000)

        self.assertFalse(result.connected)
        self.assertEqual(result.sent_frames, 0)
        self.assertEqual(result.last_frame, frame)
        with TemporaryDirectory() as directory:
            saved = save_camera_photo(result.last_frame, directory)
            self.assertEqual(saved.read_bytes(), frame)
            self.assertEqual(saved.parent, Path(directory).resolve())

    def test_second_concurrent_preview_returns_busy_without_interleaving(self):
        entered = threading.Event()
        release = threading.Event()
        frame = _source_jpeg()

        class BlockingProvider:
            def capture_stream(self, **kwargs):
                entered.set()
                release.wait(timeout=1.0)
                kwargs["on_frame"](frame)
                return frame

        first_result = []
        worker = threading.Thread(
            target=lambda: first_result.append(
                self.server.send_camera_preview(BlockingProvider(), duration_ms=3000)
            )
        )
        worker.start()
        self.assertTrue(entered.wait(timeout=1.0))

        second = self.server.send_camera_preview(_FakeFrameProvider(frame), duration_ms=3000)
        release.set()
        worker.join(timeout=1.0)

        self.assertTrue(second.busy)
        self.assertEqual(len(first_result), 1)


if __name__ == "__main__":
    unittest.main()
