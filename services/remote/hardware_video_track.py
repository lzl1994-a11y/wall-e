"""Preencoded track with bounded input, one worker and explicit teardown."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import threading
import time

from av import Packet
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import SessionDescription

from services.remote.x3_video import HardwareVideoError, require_codec_library
from services.remote.isolated_video_codec import IsolatedX3VideoCodec as X3VideoCodec


def hardware_h264_offered(sdp: str) -> bool:
    for media in SessionDescription.parse(sdp).media:
        if media.kind == "video":
            return any(c.mimeType.lower() == "video/h264"
                       and c.parameters.get("profile-level-id", "").lower() == "42001f"
                       and str(c.parameters.get("packetization-mode")) == "1"
                       for c in media.rtp.codecs)
    return False


class HardwareCameraVideoTrack(VideoStreamTrack):
    def __init__(self, gateway, library: Path):
        super().__init__()
        self._gateway = gateway
        self._library = library
        self._codec = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="walle-x3-video")
        self._closing = threading.Event()
        self._close_future = None
        self._started_at = time.monotonic()
        self._next_at = self._started_at
        self._pts = -1
        self._reported = False
        self.profile = "normal"

    @classmethod
    def create(cls, gateway):
        root = Path(__file__).resolve().parents[2]
        return cls(gateway, require_codec_library(root))

    def _encode(self, jpeg, width, height, fps, pts):
        if self._closing.is_set():
            raise MediaStreamError
        if self._codec is None:
            self._codec = X3VideoCodec(self._library)
        return self._codec.encode(jpeg, width, height, fps, pts)

    async def recv(self):
        if self.readyState != "live" or self._closing.is_set():
            raise MediaStreamError
        try:
            now = time.monotonic()
            await asyncio.sleep(max(0, self._next_at - now))
            # Startup may need camera enumeration. Never encode a previous call's frame.
            jpeg, captured_at = self._gateway.latest_raw_camera_frame()
            while not jpeg and time.monotonic() - self._started_at < 45:
                if self._closing.is_set():
                    raise MediaStreamError
                await asyncio.sleep(.05)
                jpeg, captured_at = self._gateway.latest_raw_camera_frame()
            if not jpeg or time.monotonic() - captured_at > 1.5:
                raise HardwareVideoError("camera input missing/stalled")
            width, height, fps = (320, 240, 5) if self.profile == "low" else (480, 360, 10)
            now = time.monotonic()
            self._next_at = max(self._next_at + 1 / fps, now)
            pts = max(self._pts + 1, round((now - self._started_at) * 90000))
            self._pts = pts
            data = await asyncio.get_running_loop().run_in_executor(
                self._worker, self._encode, jpeg, width, height, fps, pts
            )
            if self._closing.is_set():
                raise MediaStreamError
            packet = Packet(data)
            packet.pts = packet.dts = pts
            packet.time_base = Fraction(1, 90000)
            if not self._reported:
                self._reported = True
                self._gateway.node.get_logger().info("WebRTC 视频: X3 JPU -> NV12 -> VPU H264 Baseline，硬件上下翻转")
            return packet
        except (HardwareVideoError, OSError, ValueError) as exc:
            self._gateway.report_hardware_video_error(exc)
            self.stop()
            raise MediaStreamError from exc

    def _close_codec(self):
        if self._codec is not None:
            self._codec.close()
            self._codec = None

    def stop(self):
        if self._close_future is None:
            self._closing.set()
            # Cancellation of recv cannot interrupt an SDK call. Queue cleanup
            # behind it on the SAME worker, rather than freeing in-use memory.
            self._close_future = self._worker.submit(self._close_codec)
            self._worker.shutdown(wait=False)
        super().stop()

    async def aclose(self):
        self.stop()
        await asyncio.shield(asyncio.wrap_future(self._close_future))
