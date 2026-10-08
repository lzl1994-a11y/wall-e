"""One X3 codec subprocess; binary frames over a private, bounded socket pair.

Video scaling/SDK work must not compete for the media loop's Python lock.
The track still owns one serial worker, and closes this process after its
pending encode finishes. There is no image file, restart loop or SDK patch.
"""
from pathlib import Path
import os
import signal
import socket
import struct
import subprocess
import sys
import threading

from services.remote.x3_video import X3VideoCodec, HardwareVideoError, MAX_JPEG_BYTES

_REQUEST = struct.Struct("!IIIIq")
_RESPONSE = struct.Struct("!iI")
_owner = threading.Lock()


def _read_exact(connection, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = connection.recv(length - len(chunks))
        if not part:
            raise EOFError("X3 codec process closed its transport")
        chunks.extend(part)
    return bytes(chunks)


class IsolatedX3VideoCodec:
    def __init__(self, library: Path):
        if not _owner.acquire(blocking=False):
            raise HardwareVideoError("X3 video codec already in use")
        self._process = None
        self._connection = None
        self._closed = False
        local = child = None
        try:
            local, child = socket.socketpair()
            local.settimeout(8)
            self._connection = local
            root = Path(__file__).resolve().parents[2]
            self._process = subprocess.Popen(
                [sys.executable, "-m", "services.remote.isolated_video_codec",
                 str(library.resolve()), str(child.fileno()), str(os.getpid())],
                pass_fds=(child.fileno(),), cwd=root, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
        except BaseException:
            self.close()
            raise
        finally:
            if child is not None:
                child.close()

    def _response(self):
        status, length = _RESPONSE.unpack(_read_exact(self._connection, _RESPONSE.size))
        if length > MAX_JPEG_BYTES:
            raise HardwareVideoError("X3 codec process returned an oversized frame")
        payload = _read_exact(self._connection, length)
        if status:
            raise HardwareVideoError(payload.decode("utf-8", errors="replace"))
        return payload

    def encode(self, jpeg, width, height, fps, pts):
        if self._closed:
            raise HardwareVideoError("X3 video codec is closed")
        if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("invalid JPEG size")
        try:
            self._connection.sendall(_REQUEST.pack(len(jpeg), width, height, fps, pts))
            self._connection.sendall(jpeg)
            data = self._response()
            if not data:
                raise HardwareVideoError("X3 codec process returned an empty frame")
            return data
        except (OSError, EOFError, struct.error) as exc:
            raise HardwareVideoError(f"X3 codec process transport failed: {exc}") from exc

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = None
        try:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._connection.sendall(_REQUEST.pack(0, 0, 0, 0, 0))
                    self._response()
                except (OSError, EOFError, HardwareVideoError) as exc:
                    failure = exc
                    self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
                    failure = HardwareVideoError("X3 codec process did not exit after close")
            elif self._process is not None and self._process.returncode:
                failure = HardwareVideoError(f"X3 codec process exited: {self._process.returncode}")
        finally:
            if self._connection is not None:
                self._connection.close()
            _owner.release()
        if failure is not None:
            raise HardwareVideoError(f"X3 codec process close failed: {failure}") from failure


def _run_worker(library, descriptor, parent):
    # Do not orphan a hardware owner if the gateway is killed during an encode.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not set codec parent-death signal")
    if os.getppid() != parent:
        raise RuntimeError("X3 codec parent exited during startup")
    connection = socket.socket(fileno=descriptor)
    codec = None
    try:
        while True:
            length, width, height, fps, pts = _REQUEST.unpack(_read_exact(connection, _REQUEST.size))
            try:
                if length == 0:
                    if codec is not None:
                        codec.close()
                        codec = None
                    connection.sendall(_RESPONSE.pack(0, 0))
                    return
                if length > MAX_JPEG_BYTES:
                    raise ValueError("X3 codec request is oversized")
                jpeg = _read_exact(connection, length)
                if codec is None:
                    codec = X3VideoCodec(Path(library))
                data = codec.encode(jpeg, width, height, fps, pts)
                connection.sendall(_RESPONSE.pack(0, len(data)) + data)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}".encode("utf-8")[:4096]
                connection.sendall(_RESPONSE.pack(-1, len(message)) + message)
                return
    finally:
        try:
            if codec is not None:
                codec.close()
        finally:
            connection.close()


if __name__ == "__main__":
    _run_worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
