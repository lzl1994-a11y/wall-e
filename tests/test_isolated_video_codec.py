from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from services.remote.isolated_video_codec import (
    IsolatedX3VideoCodec, HardwareVideoError, _RESPONSE, _read_exact, _owner,
)
from services.remote.x3_video import MAX_JPEG_BYTES


def codec_with_response(data):
    codec = IsolatedX3VideoCodec.__new__(IsolatedX3VideoCodec)
    codec._closed = False
    codec._failed = False
    response = bytearray(data)
    def receive(count):
        part = bytes(response[:min(count, 3)])
        del response[:len(part)]
        return part
    codec._connection = Mock(recv=Mock(side_effect=receive))
    return codec


def test_fragmented_binary_response_is_reassembled():
    codec = codec_with_response(_RESPONSE.pack(0, 6) + b"packet")
    assert codec.encode(b"jpeg", 480, 360, 10, 9000) == b"packet"


def test_child_eof_is_explicit_and_does_not_retry():
    codec = codec_with_response(b"")
    with pytest.raises(HardwareVideoError, match="closed its transport"):
        codec.encode(b"jpeg", 480, 360, 10, 0)
    assert codec._failed
    assert codec._connection.sendall.call_count == 2


def test_oversized_response_is_rejected_before_reading_payload():
    codec = codec_with_response(_RESPONSE.pack(0, MAX_JPEG_BYTES + 1))
    with pytest.raises(HardwareVideoError, match="oversized"):
        codec.encode(b"jpeg", 480, 360, 10, 0)


def test_native_error_is_reported_to_track():
    message = b"X3 video JPEG decode failed: -17"
    codec = codec_with_response(_RESPONSE.pack(-1, len(message)) + message)
    with pytest.raises(HardwareVideoError, match="decode failed: -17"):
        codec.encode(b"jpeg", 480, 360, 10, 0)
    assert codec._failed


def test_failed_launch_closes_sockets_and_releases_owner():
    local, child = Mock(), Mock(fileno=Mock(return_value=17))
    with patch("services.remote.isolated_video_codec.socket.socketpair", return_value=(local, child)), patch(
        "services.remote.isolated_video_codec.subprocess.Popen", side_effect=OSError("launch failed")
    ):
        with pytest.raises(OSError, match="launch failed"):
            IsolatedX3VideoCodec(Path("unused"))
    local.close.assert_called_once()
    child.close.assert_called_once()
    assert _owner.acquire(blocking=False)
    _owner.release()


def test_child_crash_after_close_ack_is_reported_and_owner_is_released():
    codec = codec_with_response(_RESPONSE.pack(0, 0))
    codec._closed = False
    codec._failed = False
    codec._process = Mock(poll=Mock(return_value=None), returncode=None)
    codec._process.wait.side_effect = lambda **kwargs: setattr(codec._process, "returncode", 17)
    assert _owner.acquire(blocking=False)
    with pytest.raises(HardwareVideoError, match="exited: 17"):
        codec.close()
    codec._connection.close.assert_called_once()
    assert _owner.acquire(blocking=False)
    _owner.release()
