#!/usr/bin/env python3
"""WebRTC robot gateway for the Walle remote-control web client.

The gateway keeps the browser-facing transport separate from ROS. Signaling is
only used for SDP/ICE exchange; reliable commands and expiring motion use
separate RTCDataChannels, with media carried by WebRTC tracks.
"""

from __future__ import annotations

import asyncio
from array import array
import json
import os
import threading
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import MultiArrayDimension, String, UInt8MultiArray

from aiortc import (
    AudioStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.sdp import candidate_from_sdp
from aiortc.rtcicetransport import parse_stun_turn_uri
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler, VideoFrame

from services.action.action_command import ACTION_REQUEST_TOPIC, new_action_request_id
from services.audio.audio_control_protocol import (
    AUDIO_CONTROL_TOPIC,
    encode_stop_speech_control,
)
from services.motion.motor_control import mix_differential_drive
from services.motion.motion_arbiter import MOTOR_REMOTE_TOPIC, STOP_COMMAND
from services.motion.servo_motion_config import load_neck_kinematics
from services.remote.remote_protocol import (
    REMOTE_AUDIO_PLAYBACK_TOPIC,
    REMOTE_INTERCOM_STATE_TOPIC,
    REMOTE_SAFETY_SOURCE,
    REMOTE_SOURCE,
    ROBOT_AUDIO_PCM_TOPIC,
    ROBOT_MIC_SAMPLE_RATE,
    action_request_for,
    encode_call_state,
    decode_remote_message,
    encode_voice_state,
)
from services.remote.realtime_control import ChangedTargetGate, ChannelSequenceGate
from services.remote.audio_playback_gate import REMOTE_AUDIO_EPOCH_LABEL_PREFIX
from services.remote.hardware_video_track import HardwareCameraVideoTrack, hardware_h264_offered
from services.vision.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    CAMERA_SOURCE_TOPIC,
    RAW_CAMERA_CLIENT_PREFIX,
    CAMERA_STATUS_TOPIC,
    encode_camera_command,
)


SIGNALING_RECONNECT_DELAY_SEC = 1.0
SIGNALING_RECONNECT_MAX_SEC = 30.0
CAMERA_LEASE_SEC = 15.0
CAMERA_RENEW_SEC = 5.0
ROBOT_AUDIO_SAMPLE_RATE = 48_000
ROBOT_AUDIO_FRAME_SAMPLES = 960
VIDEO_CLOCK_RATE = 90_000
VIDEO_FPS = 10
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360


def _safe_signaling_url(url: str) -> str:
    """Keep query/fragment credentials out of diagnostic logs."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        return "<invalid-signaling-url>"


def _ice_servers(config: dict) -> list[RTCIceServer]:
    """Build configurable ICE servers without embedding deployment secrets."""
    raw = os.environ.get("WALLE_WEBRTC_ICE_SERVERS")
    if raw:
        try:
            raw = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raw = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        raw = config.get("ice_servers", [])
    if not isinstance(raw, list):
        raw = [raw]

    servers = []
    for item in raw:
        username = None
        credential = None
        if isinstance(item, str):
            urls = item.strip()
        elif isinstance(item, dict):
            urls = item.get("urls", item.get("url", ""))
            username = item.get("username")
            credential = item.get("credential")
        else:
            continue
        if isinstance(urls, str):
            urls = urls.strip()
        elif isinstance(urls, list):
            urls = [url for url in urls if isinstance(url, str) and url.strip()]
        else:
            continue
        if not urls:
            continue
        try:
            servers.append(
                RTCIceServer(
                    urls=urls,
                    username=username if isinstance(username, str) else None,
                    credential=credential if isinstance(credential, str) else None,
                )
            )
        except (TypeError, ValueError):
            continue
    return servers


def _repository_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "core" / "config.yaml"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _ice_servers_for_transport(servers: list[RTCIceServer], transport=None):
    """aiortc uses ONE TURN URL. A retry must explicitly choose another one."""
    if transport is None:
        return list(servers)
    if transport != "tcp":
        raise ValueError("不支持的 TURN 备用传输类型")
    stun = []
    selected = None
    for server in servers:
        for url in server.urls if isinstance(server.urls, list) else [server.urls]:
            try:
                parsed = parse_stun_turn_uri(url)
            except (ValueError, TypeError):
                continue
            if parsed["scheme"] == "stun" and not stun:
                stun.append(RTCIceServer(urls=url))
            if (selected is None and parsed["scheme"] in {"turn", "turns"}
                    and parsed["transport"] == "tcp" and server.credentialType == "password"):
                selected = RTCIceServer(
                    urls=url, username=server.username, credential=server.credential,
                    credentialType=server.credentialType,
                )
    if selected is None:
        raise ValueError("瓦力端未配置可用的 TURN TCP 备用地址")
    return stun + [selected]


def _remote_config() -> dict:
    config = _repository_config().get("webrtc_remote", {})
    return config if isinstance(config, dict) else {}


def _resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    if source_rate <= 0 or source_rate == target_rate or samples.size <= 1:
        return samples
    count = max(1, round(samples.size * target_rate / source_rate))
    positions = np.linspace(0, samples.size - 1, count)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.int16)


class CameraVideoTrack(VideoStreamTrack):
    """Convert the existing leased JPEG topic into a WebRTC video track."""

    def __init__(self, gateway: "RemoteWebRtcGateway"):
        super().__init__()
        self._gateway = gateway
        self._next_frame_at = time.monotonic()
        self._started_at = self._next_frame_at
        self._pts = 0
        self.profile = "normal"

    async def recv(self):
        now = time.monotonic()
        if self._next_frame_at > now:
            await asyncio.sleep(self._next_frame_at - now)
        width, height, fps = (320, 240, 5) if self.profile == "low" else (VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS)
        self._next_frame_at += 1.0 / fps
        if self._next_frame_at < time.monotonic():
            self._next_frame_at = time.monotonic()
        # Frame-count timestamps fall behind wall time when an encoder stalls,
        # making browser A/V synchronisation add growing audio delay.
        pts = max(self._pts, round((time.monotonic() - self._started_at) * VIDEO_CLOCK_RATE))
        self._pts = pts + 1
        time_base = Fraction(1, VIDEO_CLOCK_RATE)
        jpeg = self._gateway.latest_camera_frame()
        try:
            # JPEG decode, resize and colorspace conversion are CPU work. Keep
            # them off aiortc's event loop so camera frames cannot stall audio,
            # ICE timers or data-channel control.
            frame = await asyncio.to_thread(self._make_frame, jpeg, width, height)
        except (OSError, ValueError) as exc:
            self._gateway.report_camera_error(exc)
            frame = await asyncio.to_thread(self._make_frame, b"", width, height)
        frame.pts = pts
        frame.time_base = time_base
        return frame

    @staticmethod
    def _make_frame(jpeg: bytes, width=VIDEO_WIDTH, height=VIDEO_HEIGHT) -> VideoFrame:
        if not jpeg:
            image = Image.new("RGB", (width, height), (4, 15, 19))
        else:
            with Image.open(BytesIO(jpeg)) as source:
                # Let the native JPEG decoder discard unneeded DCT detail.
                # draft() must precede convert()/load(); decoding the full
                # camera resolution wastes CPU on pixels we immediately resize.
                source.draft("RGB", (width, height))
                image = source.convert("RGB")
        # VP8/H264 need YUV420. Use the existing native scaler for resizing
        # and colorspace conversion together, rather than resizing RGB in
        # Pillow and converting it again in the encoder worker.
        return VideoFrame.from_image(image).reformat(
            width=width, height=height, format="yuv420p", interpolation="BILINEAR",
        )


class RobotAudioTrack(AudioStreamTrack):
    """Pace the robot's existing 48 kHz PCM output into WebRTC audio frames."""

    def __init__(self):
        super().__init__()
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._pts = 0
        self._next_frame_at = time.monotonic()
        self._started_at = self._next_frame_at

    def push(self, pcm: bytes, sample_rate: int = ROBOT_AUDIO_SAMPLE_RATE) -> None:
        if not pcm:
            return
        raw = pcm[: len(pcm) - len(pcm) % 2]
        if not raw:
            return
        samples = np.frombuffer(raw, dtype=np.int16)
        samples = _resample_mono(samples, sample_rate, ROBOT_AUDIO_SAMPLE_RATE)
        with self._lock:
            self._buffer.extend(samples.tobytes())
            max_bytes = int(ROBOT_AUDIO_SAMPLE_RATE * 2 * 0.2)
            if len(self._buffer) > max_bytes:
                del self._buffer[:-max_bytes]

    async def recv(self):
        frame_duration = ROBOT_AUDIO_FRAME_SAMPLES / ROBOT_AUDIO_SAMPLE_RATE
        now = time.monotonic()
        if self._next_frame_at > now:
            await asyncio.sleep(self._next_frame_at - now)
        elif now - self._next_frame_at > frame_duration * 2:
            self._next_frame_at = now
            self._pts = max(self._pts, round((now - self._started_at) * ROBOT_AUDIO_SAMPLE_RATE))
        self._next_frame_at += frame_duration
        needed = ROBOT_AUDIO_FRAME_SAMPLES * 2
        with self._lock:
            if len(self._buffer) > needed * 4:
                del self._buffer[: len(self._buffer) - needed * 2]
            payload = bytes(self._buffer[:needed])
            del self._buffer[:len(payload)]
        payload = payload.ljust(needed, b"\x00")
        frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=ROBOT_AUDIO_FRAME_SAMPLES,
        )
        frame.planes[0].update(payload)
        frame.sample_rate = ROBOT_AUDIO_SAMPLE_RATE
        frame.time_base = Fraction(1, ROBOT_AUDIO_SAMPLE_RATE)
        frame.pts = self._pts
        self._pts += ROBOT_AUDIO_FRAME_SAMPLES
        return frame


class RemoteWebRtcGateway:
    def __init__(
        self,
        node: "RemoteWebRtcNode",
        *,
        signaling_url: str,
        robot_id: str,
        token: str = "",
        servo_step_size: float = 50.0,
        ice_servers: list[RTCIceServer] | None = None,
        video_backend: str = "software",
    ):
        self.node = node
        self.signaling_url = signaling_url
        self.robot_id = robot_id
        self.token = token
        if video_backend not in {"software", "x3"}:
            raise ValueError("webrtc_remote.video_backend must be software or x3")
        self._video_backend = video_backend
        self.servo_step_size = max(0.1, min(65535.0, float(servo_step_size)))
        self._ice_servers = list(ice_servers or [])
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop_event: asyncio.Event | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None
        self._controller_peer_id: str | None = None
        self._negotiation_id: int | None = None
        self._peer: RTCPeerConnection | None = None
        self._channel = None
        self._video_track: CameraVideoTrack | None = None
        self._audio_track: RobotAudioTrack | None = None
        self._camera_renew_task: asyncio.Task | None = None
        self._audio_tasks: set[asyncio.Task] = set()
        self._last_jpeg = b""
        self._last_raw_jpeg = b""
        self._raw_frame_at = 0.0
        self._frame_lock = threading.Lock()
        self._camera_error_reported = False
        self._sequence_gate = ChannelSequenceGate()
        self._channels = {}
        self._servo_gate = ChangedTargetGate(min_interval_sec=0.1)
        self._remote_voice_active = False
        self._remote_call_active = False
        self._remote_audio_epoch = 0
        self._camera_client_id = f"webrtc:{robot_id}"[:96]
        self._neck_kinematics = load_neck_kinematics()

        self._camera_command_pub = node.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        self._motor_pub = node.create_publisher(String, MOTOR_REMOTE_TOPIC, 10)
        self._action_pub = node.create_publisher(String, ACTION_REQUEST_TOPIC, 10)
        self._remote_audio_pub = node.create_publisher(
            UInt8MultiArray, REMOTE_AUDIO_PLAYBACK_TOPIC, qos_profile_sensor_data
        )
        self._intercom_state_pub = node.create_publisher(
            String, REMOTE_INTERCOM_STATE_TOPIC, 10
        )
        self._audio_control_pub = node.create_publisher(String, AUDIO_CONTROL_TOPIC, 10)
        self._camera_sub = node.create_subscription(
            CompressedImage,
            CAMERA_FRAME_TOPIC,
            self._on_camera_frame,
            qos_profile_sensor_data,
        )
        self._raw_camera_sub = node.create_subscription(
            CompressedImage, CAMERA_SOURCE_TOPIC, self._on_raw_camera_frame, qos_profile_sensor_data
        )
        self._camera_status_sub = node.create_subscription(
            String,
            CAMERA_STATUS_TOPIC,
            self._on_camera_status,
            10,
        )
        self._robot_audio_sub = node.create_subscription(
            UInt8MultiArray,
            ROBOT_AUDIO_PCM_TOPIC,
            self._on_robot_audio,
            qos_profile_sensor_data,
        )

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_thread,
            name="walle-webrtc-gateway",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._request_async_stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._stop_robot_outputs()

    def _request_async_stop(self) -> None:
        if self._async_stop_event is not None:
            self._async_stop_event.set()
        socket = self._socket
        if socket is not None:
            asyncio.create_task(socket.close())

    async def _sleep_or_stop(self, delay: float) -> None:
        event = self._async_stop_event
        if event is None:
            await asyncio.sleep(delay)
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    def latest_camera_frame(self) -> bytes:
        with self._frame_lock:
            return self._last_jpeg

    def report_camera_error(self, exc: Exception) -> None:
        """Keep bad JPEG data local to the track and report it once."""
        if self._camera_error_reported:
            return
        self._camera_error_reported = True
        self.node.get_logger().warning(f"WebRTC 摄像头帧异常，发送占位帧: {exc}")
        self._send_event_threadsafe({
            "type": "warning",
            "message": "摄像头帧异常，当前发送占位画面",
        })

    def latest_raw_camera_frame(self) -> tuple[bytes, float]:
        with self._frame_lock:
            return self._last_raw_jpeg, self._raw_frame_at

    def _on_raw_camera_frame(self, message: CompressedImage) -> None:
        # One latest slot: no ROS input backlog is fed to the encoder.
        if self._video_backend == "x3" and message.data:
            with self._frame_lock:
                self._last_raw_jpeg = bytes(message.data)
                self._raw_frame_at = time.monotonic()

    def report_hardware_video_error(self, exc: Exception) -> None:
        self.node.get_logger().error(f"WebRTC 硬件视频失败，关闭会话: {exc}")
        self._send_event_threadsafe({"type": "error", "message": "硬件视频失败，连接已进入安全状态"})
        self._stop_robot_outputs()
        self._schedule_peer_close()

    def _on_camera_frame(self, message: CompressedImage) -> None:
        jpeg = bytes(message.data or b"")
        if jpeg:
            with self._frame_lock:
                self._last_jpeg = jpeg

    def _on_camera_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("state") != "error":
            return
        error = str(payload.get("error") or "摄像头状态异常")[:200]
        self.node.get_logger().error(f"WebRTC 摄像头不可用: {error}")
        self._send_event_threadsafe({
            "type": "error",
            "message": "摄像头不可用，连接已进入安全状态",
        })
        if self._peer is not None:
            self._stop_robot_outputs()
            self._schedule_peer_close()

    def _on_robot_audio(self, message: UInt8MultiArray) -> None:
        track = self._audio_track
        if track is not None:
            track.push(
                bytes(message.data or b""),
                sample_rate=ROBOT_MIC_SAMPLE_RATE,
            )

    def _run_thread(self) -> None:
        try:
            asyncio.run(self._run_loop())
        except Exception as exc:
            self.node.get_logger().error(f"WebRTC 网关退出: {exc}")

    async def _run_loop(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            self.node.get_logger().error(
                f"缺少 WebRTC 网关依赖 websockets/aiortc: {exc}"
            )
            return

        self._loop = asyncio.get_running_loop()
        self._async_stop_event = asyncio.Event()
        reconnect_delay = SIGNALING_RECONNECT_DELAY_SEC
        while not self._stopping.is_set():
            connected_at = time.monotonic()
            try:
                async with websockets.connect(
                    self.signaling_url,
                    max_size=128 * 1024,
                    ping_interval=20,
                    open_timeout=10,
                ) as socket:
                    self._socket = socket
                    self.node.get_logger().info(
                        f"WebSocket 信令已连接: {_safe_signaling_url(self.signaling_url)}"
                    )
                    await socket.send(json.dumps({
                        "type": "join",
                        "roomId": self.robot_id,
                        "role": "robot",
                        "token": self.token or None,
                    }, separators=(",", ":")))
                    async for raw in socket:
                        await self._handle_signaling(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._stopping.is_set():
                    self.node.get_logger().warning(
                        f"信令连接断开，将在 {reconnect_delay:.1f}s 后重试: {exc}"
                    )
            finally:
                await self._close_peer()
                self._socket = None
                if time.monotonic() - connected_at >= 5.0:
                    reconnect_delay = SIGNALING_RECONNECT_DELAY_SEC
            if not self._stopping.is_set():
                await self._sleep_or_stop(reconnect_delay)
                reconnect_delay = min(
                    SIGNALING_RECONNECT_MAX_SEC,
                    reconnect_delay * 2.0,
                )

    async def _handle_signaling(self, raw) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return
        if message.get("type") == "signal":
            sender = message.get("from")
            if isinstance(sender, str):
                if (
                    self._peer is not None
                    and self._controller_peer_id
                    and sender != self._controller_peer_id
                ):
                    self.node.get_logger().warning(
                        "拒绝第二个远程控制端的信令请求"
                    )
                    return
                self._controller_peer_id = sender
            signal = message.get("signal")
            if isinstance(signal, dict):
                await self._handle_signal(signal, sender=sender)
        elif message.get("type") == "peer-left":
            peer_id = message.get("peerId")
            if not peer_id or peer_id == self._controller_peer_id:
                await self._close_peer()
        elif message.get("type") == "peer-ready" and message.get("role") == "controller":
            peer_id = str(message.get("peerId") or "") or None
            if self._peer is None or peer_id == self._controller_peer_id:
                self._controller_peer_id = peer_id
            else:
                self.node.get_logger().warning(
                    "已有远程控制端连接，忽略新的 controller"
                )

    async def _handle_signal(self, signal: dict, *, sender: str | None = None) -> None:
        kind = signal.get("type")
        if kind == "offer" and isinstance(signal.get("sdp"), str):
            negotiation_id = signal.get("negotiationId")
            transport = signal.get("turnTransport")
            if (negotiation_id is not None and (isinstance(negotiation_id, bool)
                    or not isinstance(negotiation_id, int) or not 0 <= negotiation_id <= 2**53 - 1)):
                return
            if transport not in (None, "tcp"):
                return
            await self._accept_offer(
                signal["sdp"], sender=sender, negotiation_id=negotiation_id,
                turn_transport=transport,
            )
        elif kind == "candidate":
            await self._accept_candidate(signal)

    async def _accept_offer(self, sdp: str, *, sender: str | None = None,
                            negotiation_id: int | None = None, turn_transport=None) -> None:
        controller_peer_id = sender or self._controller_peer_id
        if self._peer is not None:
            if not sender or sender != self._controller_peer_id:
                self.node.get_logger().warning("忽略第二个 controller 的 offer")
                return
        try:
            servers = _ice_servers_for_transport(self._ice_servers, turn_transport)
        except ValueError as exc:
            self.node.get_logger().warning(str(exc))
            if self._socket is not None and controller_peer_id:
                await self._socket.send(json.dumps({
                    "type": "signal", "to": controller_peer_id,
                    "signal": {"type": "error", "message": str(exc), "negotiationId": negotiation_id},
                }, separators=(",", ":")))
            return
        await self._close_peer()
        # _close_peer() clears the old session identity. Preserve the sender
        # of this offer so the answer can be routed back through signaling.
        self._controller_peer_id = controller_peer_id
        self._negotiation_id = negotiation_id
        peer = RTCPeerConnection(
            RTCConfiguration(iceServers=servers)
        )
        self._peer = peer
        self.node.get_logger().info(
            f"PeerConnection 建立: controller={self._controller_peer_id or '<unknown>'}"
        )
        if turn_transport:
            self.node.get_logger().info("初次 ICE 失败后明确选择 TURN TCP 备用配置")
        use_hardware = self._video_backend == "x3" and hardware_h264_offered(sdp)
        with self._frame_lock:
            self._last_raw_jpeg = b""
            self._raw_frame_at = 0.0
        self._camera_client_id = f"{RAW_CAMERA_CLIENT_PREFIX if use_hardware else 'webrtc:'}{self.robot_id}"[:96]
        self._video_track = HardwareCameraVideoTrack.create(self) if use_hardware else CameraVideoTrack(self)
        if self._video_backend == "x3" and not use_hardware:
            self.node.get_logger().info("浏览器未提供 H264 Baseline 42001f，选择兼容软件视频")
        self._audio_track = RobotAudioTrack()
        peer.addTrack(self._video_track)
        peer.addTrack(self._audio_track)
        self._acquire_camera()

        @peer.on("datachannel")
        def on_datachannel(channel):
            if self._peer is not peer:
                channel.close()
                return
            if channel.label not in {"control", "motion"} or channel.label in self._channels:
                channel.close()
                return
            self._channels[channel.label] = channel
            if channel.label == "control":
                self._channel = channel

            @channel.on("open")
            def on_open():
                if self._peer is not peer:
                    return
                self._send_event({
                    "type": "status",
                    "value": "机器人 WebRTC 网关已就绪",
                })
                self.node.get_logger().info("WebRTC DataChannel 已打开")

            @channel.on("message")
            def on_message(message):
                if self._peer is peer:
                    self._on_data_message(message, channel.label)

            @channel.on("close")
            def on_close():
                if self._peer is not peer:
                    return
                self._sequence_gate.stopped = True
                self._stop_robot_outputs()
                if self._loop and self._loop.is_running():
                    self._loop.create_task(self._close_peer(peer))

        @peer.on("track")
        def on_track(track):
            if self._peer is peer and track.kind == "audio":
                task = asyncio.create_task(self._consume_remote_audio(track))
                self._audio_tasks.add(task)
                task.add_done_callback(self._audio_tasks.discard)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            self.node.get_logger().info(
                f"PeerConnection 状态: {peer.connectionState}"
            )
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await self._close_peer(peer)

        if use_hardware:
            # aiortc applies these preferences while processing the offer.
            # Applying them afterwards leaves VP8 first in the answer, although
            # this track already supplies encoded H264 packets.
            codecs = [c for c in RTCRtpSender.getCapabilities("video").codecs
                      if c.mimeType == "video/H264" and c.parameters.get("profile-level-id") == "42001f"]
            for transceiver in peer.getTransceivers():
                if transceiver.kind == "video":
                    transceiver.setCodecPreferences(codecs)
        await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
        await self._wait_ice_complete(peer)
        local = peer.localDescription
        if self._peer is peer and local and self._socket is not None and self._controller_peer_id:
            await self._socket.send(json.dumps({
                "type": "signal",
                "to": self._controller_peer_id,
                "signal": {"type": "answer", "sdp": local.sdp,
                           "negotiationId": negotiation_id, "turnTransport": turn_transport},
            }, separators=(",", ":")))

    async def _wait_ice_complete(self, peer: RTCPeerConnection) -> None:
        deadline = time.monotonic() + 8.0
        while peer.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def _accept_candidate(self, signal: dict) -> None:
        if signal.get("negotiationId") != self._negotiation_id:
            return
        peer = self._peer
        value = signal.get("candidate")
        if peer is None or not isinstance(value, str) or not value:
            return
        try:
            candidate = candidate_from_sdp(value.removeprefix("candidate:"))
            candidate.sdpMid = signal.get("sdpMid")
            candidate.sdpMLineIndex = signal.get("sdpMLineIndex")
            await peer.addIceCandidate(candidate)
        except (TypeError, ValueError, AttributeError) as exc:
            self.node.get_logger().warning(f"忽略无效 ICE candidate: {exc}")
            self._send_event({
                "type": "warning",
                "message": "忽略无效 ICE candidate",
            })

    def _on_data_message(self, raw, channel="control") -> None:
        message = decode_remote_message(raw)
        if message is None:
            self.node.get_logger().warning("收到非法远程控制消息，进入安全状态")
            self._send_event({
                "type": "warning",
                "message": "非法远程控制消息，连接已进入安全状态",
            })
            self._stop_robot_outputs()
            self._schedule_peer_close()
            return
        try:
            if not self._sequence_gate.accept(channel, message):
                return
        except ValueError as exc:
            self.node.get_logger().warning(f"拒绝远程控制消息: {exc}")
            self._send_event({
                "type": "warning",
                "message": "远程控制序号重复或倒退，连接已进入安全状态",
            })
            self._stop_robot_outputs()
            self._schedule_peer_close()
            return
        message_type = message["type"]
        if message_type == "control":
            self._publish_control(message["vector"])
        elif message_type == "stop":
            self._stop_robot_outputs()
            self._schedule_peer_close()
        elif message_type == "action":
            self._publish_action(message["name"])
        elif message_type == "voice":
            self._set_remote_voice(message["state"] == "start")
        elif message_type == "call":
            self._set_remote_call(message["state"] == "start")
        elif message_type == "media" and self._video_track is not None:
            self._video_track.profile = message["video"]
            self._send_event({"type": "status", "value": f"视频档位：{message['video']}"})

    def _publish_control(self, vector: dict[str, float]) -> None:
        forward = float(vector["forward"])
        turn = float(vector["turn"])
        self._motor_pub.publish(String(data=json.dumps(
            mix_differential_drive(forward, turn),
            separators=(",", ":"),
        )))

        yaw = float(vector["yaw"])
        pitch = float(vector["pitch"])
        targets = {"head_yaw": int(5000 - yaw * 2600)}
        targets.update(self._neck_kinematics.targets(-pitch))
        if not self._servo_gate.accept(targets, time.monotonic()):
            return
        self._action_pub.publish(String(data=json.dumps({
            "name": "manual_servo",
            "arguments": {
                "targets": targets,
                "step_size": self.servo_step_size,
            },
            "request_id": new_action_request_id(),
            "source": REMOTE_SOURCE,
        }, ensure_ascii=False, separators=(",", ":"))))

    def _publish_action(self, name: str) -> None:
        # The action coordinator accepts the canonical play_sequence skill;
        # browser-facing names are intentionally only sequence-name aliases.
        action_name, arguments = action_request_for(name)
        self._action_pub.publish(String(data=json.dumps({
            "name": action_name,
            "arguments": arguments,
            "request_id": new_action_request_id(),
            "source": REMOTE_SOURCE,
        }, ensure_ascii=False, separators=(",", ":"))))

    def _publish_audio_control(self, source: str) -> None:
        publisher = getattr(self, "_audio_control_pub", None)
        if publisher is not None:
            publisher.publish(String(data=encode_stop_speech_control(source)))

    def _stop_robot_outputs(self) -> None:
        self._sequence_gate.stopped = True
        self._motor_pub.publish(String(data=json.dumps(STOP_COMMAND, separators=(",", ":"))))
        self._action_pub.publish(String(data=json.dumps({
            "name": "stop_all",
            "arguments": {},
            "request_id": new_action_request_id(),
            "source": REMOTE_SAFETY_SOURCE,
        }, separators=(",", ":"))))
        self._publish_audio_control("remote_safety")
        if self._remote_voice_active:
            self._set_remote_voice(False)
        if self._remote_call_active:
            self._set_remote_call(False)
        self._release_camera()

    def _schedule_peer_close(self) -> None:
        loop = self._loop
        peer = self._peer
        if loop and loop.is_running() and peer is not None:
            def close_if_current():
                if self._peer is peer:
                    loop.create_task(self._close_peer(peer))

            loop.call_soon_threadsafe(close_if_current)

    def _set_remote_voice(self, active: bool) -> None:
        if active == self._remote_voice_active:
            return
        if active and not (self._remote_voice_active or self._remote_call_active):
            self._remote_audio_epoch = max(self._remote_audio_epoch + 1, time.monotonic_ns())
        self._remote_voice_active = active
        self._intercom_state_pub.publish(
            String(data=encode_voice_state("start" if active else "stop", epoch=self._remote_audio_epoch))
        )
        self.node.get_logger().info(
            f"远程控制端上行语音{'开始' if active else '停止'}"
        )

    def _set_remote_call(self, active: bool) -> None:
        if active == self._remote_call_active:
            return
        if active and not (self._remote_voice_active or self._remote_call_active):
            self._remote_audio_epoch = max(self._remote_audio_epoch + 1, time.monotonic_ns())
        self._remote_call_active = active
        if not active:
            self._publish_audio_control("remote_call_end")
        self._intercom_state_pub.publish(
            String(data=encode_call_state("start" if active else "end", epoch=self._remote_audio_epoch))
        )
        self.node.get_logger().info(
            f"远程全双工通话{'开始' if active else '结束'}"
        )

    async def _consume_remote_audio(self, track) -> None:
        resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=ROBOT_AUDIO_SAMPLE_RATE,
        )
        while not self._stopping.is_set() and track.readyState == "live":
            try:
                frame = await track.recv()
            except MediaStreamError:
                self._remote_audio_ended("手机麦克风音轨结束")
                return
            except Exception as exc:
                self.node.get_logger().warning(f"远程音频接收失败: {exc}")
                self._remote_audio_ended("手机麦克风音频接收异常")
                return
            if not (self._remote_voice_active or self._remote_call_active):
                continue
            epoch = self._remote_audio_epoch
            try:
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1)
                    if samples.size == 0:
                        continue
                    message = UInt8MultiArray()
                    message.layout.dim = [MultiArrayDimension(
                        label=f"{REMOTE_AUDIO_EPOCH_LABEL_PREFIX}{epoch}",
                        size=int(samples.size), stride=int(samples.size),
                    )]
                    message.data = array(
                        "B",
                        samples.astype(np.int16, copy=False).tobytes(),
                    )
                    self._remote_audio_pub.publish(message)
            except (AttributeError, TypeError, ValueError) as exc:
                self.node.get_logger().warning(f"远程音频帧解析失败: {exc}")
                self._remote_audio_ended("手机麦克风音频处理异常")
                return

    def _remote_audio_ended(self, reason: str) -> None:
        """Stop the failed audio direction while preserving video and control."""
        peer = self._peer
        state = peer.connectionState if peer is not None else "no-peer"
        self.node.get_logger().warning(
            f"{reason} (PeerConnection={state})；保留视频和控制连接"
        )
        self._publish_audio_control("remote_audio_track_ended")
        self._set_remote_voice(False)
        self._set_remote_call(False)
        self._send_event({
            "type": "warning",
            "message": "手机麦克风音轨已中断；视频和遥控仍保持连接",
        })

    def _send_event_threadsafe(self, event: dict) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._send_event, event)

    def _send_event(self, event: dict) -> None:
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return
        try:
            channel.send(json.dumps(
                {"type": "event", "event": event},
                ensure_ascii=False,
                separators=(",", ":"),
            ))
        except Exception as exc:
            self.node.get_logger().warning(f"远程事件发送失败: {exc}")

    def _acquire_camera(self) -> None:
        self.node.get_logger().info("申请 WebRTC 摄像头 lease")
        self._camera_command_pub.publish(String(data=encode_camera_command(
            "acquire", self._camera_client_id, CAMERA_LEASE_SEC
        )))
        if self._loop:
            self._camera_renew_task = self._loop.create_task(self._renew_camera())

    async def _renew_camera(self) -> None:
        while self._peer is not None and not self._stopping.is_set():
            await asyncio.sleep(CAMERA_RENEW_SEC)
            if self._peer is not None:
                self._camera_command_pub.publish(String(data=encode_camera_command(
                    "renew", self._camera_client_id, CAMERA_LEASE_SEC
                )))

    def _release_camera(self) -> None:
        task = self._camera_renew_task
        self._camera_renew_task = None
        if task and not task.done():
            loop = self._loop
            if loop and loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()
        self.node.get_logger().info("释放 WebRTC 摄像头 lease")
        self._camera_command_pub.publish(String(data=encode_camera_command(
            "release", self._camera_client_id
        )))

    async def _close_peer(self, expected_peer: RTCPeerConnection | None = None) -> None:
        if expected_peer is not None and self._peer is not expected_peer:
            return
        # Detach the old session BEFORE awaiting teardown. A late close from
        # the old peer must not clear a freshly negotiated TCP retry.
        tasks = list(self._audio_tasks)
        self._audio_tasks.clear()
        peer = self._peer
        self._peer = None
        self._channel = None
        self._controller_peer_id = None
        self._negotiation_id = None
        video_track = self._video_track
        self._video_track = None
        self._audio_track = None
        self._servo_gate.reset()
        self._stop_robot_outputs()
        self._channels.clear()
        self._sequence_gate.reset()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if peer is not None:
            await peer.close()
            self.node.get_logger().info("PeerConnection 已关闭并进入安全状态")
        if isinstance(video_track, HardwareCameraVideoTrack):
            try:
                await video_track.aclose()
            except Exception as exc:
                self.node.get_logger().error(f"WebRTC 硬件资源释放失败: {exc}")


class RemoteWebRtcNode(Node):
    def __init__(self):
        super().__init__("remote_webrtc_gateway_node")
        config = _remote_config()
        signaling_value = os.environ.get(
            "WALLE_WEBRTC_SIGNALING_URL",
            config.get("signaling_url", "ws://127.0.0.1:8787/signal"),
        )
        robot_value = os.environ.get(
            "WALLE_WEBRTC_ROBOT_ID", config.get("robot_id", "WALLY-01")
        )
        token_value = os.environ.get("WALLE_WEBRTC_SIGNALING_TOKEN", config.get("token", ""))
        signaling_url = (
            signaling_value.strip()
            if isinstance(signaling_value, str) and signaling_value.strip()
            else "ws://127.0.0.1:8787/signal"
        )
        robot_id = (
            robot_value.strip()
            if isinstance(robot_value, str) and robot_value.strip()
            else "WALLY-01"
        )
        token = token_value.strip() if isinstance(token_value, str) else ""
        step_size = config.get("servo_step_size", 50.0)
        ice_servers = _ice_servers(config)
        self._gateway = RemoteWebRtcGateway(
            self,
            signaling_url=signaling_url,
            robot_id=robot_id,
            token=token,
            servo_step_size=step_size,
            ice_servers=ice_servers,
            video_backend=config.get("video_backend", "software"),
        )
        self._gateway.start()
        self.get_logger().info(
            f"WebRTC 机器人网关上线: room={robot_id}, "
            f"signaling={_safe_signaling_url(signaling_url)}, "
            f"auth={'enabled' if token else 'disabled'}, "
            f"ice_servers={len(ice_servers)}"
        )

    def destroy_node(self):
        self._gateway.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RemoteWebRtcNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
