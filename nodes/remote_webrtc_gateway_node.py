#!/usr/bin/env python3
"""WebRTC robot gateway for the Walle remote-control web client.

The gateway keeps the browser-facing transport separate from ROS. Signaling is
only used for SDP/ICE exchange; control messages arrive over an ordered
RTCDataChannel and media arrives/leaves as WebRTC tracks.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, UInt8MultiArray

from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from av import AudioFrame, VideoFrame

from services.action.action_command import ACTION_REQUEST_TOPIC, new_action_request_id
from services.motion.motor_control import mix_differential_drive
from services.motion.motion_arbiter import MOTOR_REMOTE_TOPIC, STOP_COMMAND
from services.motion.servo_motion_config import load_neck_kinematics
from services.remote.remote_protocol import (
    REMOTE_AUDIO_PCM_TOPIC,
    REMOTE_SAFETY_SOURCE,
    REMOTE_SOURCE,
    REMOTE_VOICE_STATE_TOPIC,
    decode_remote_message,
    encode_voice_state,
)
from services.vision.camera_capture_protocol import (
    CAMERA_COMMAND_TOPIC,
    CAMERA_FRAME_TOPIC,
    encode_camera_command,
)


SIGNALING_RECONNECT_DELAY_SEC = 3.0
CAMERA_LEASE_SEC = 15.0
CAMERA_RENEW_SEC = 5.0
REMOTE_AUDIO_SAMPLE_RATE = 16_000
ROBOT_AUDIO_SAMPLE_RATE = 48_000
ROBOT_AUDIO_FRAME_SAMPLES = 960
VIDEO_CLOCK_RATE = 90_000
VIDEO_FPS = 15


def _repository_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "core" / "config.yaml"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


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
        self._pts = 0

    async def recv(self):
        now = time.monotonic()
        if self._next_frame_at > now:
            await asyncio.sleep(self._next_frame_at - now)
        self._next_frame_at += 1.0 / VIDEO_FPS
        if self._next_frame_at < time.monotonic():
            self._next_frame_at = time.monotonic()
        pts = self._pts
        self._pts += VIDEO_CLOCK_RATE // VIDEO_FPS
        time_base = Fraction(1, VIDEO_CLOCK_RATE)
        jpeg = self._gateway.latest_camera_frame()
        try:
            if jpeg:
                with Image.open(BytesIO(jpeg)) as image:
                    frame = VideoFrame.from_image(image.convert("RGB"))
            else:
                frame = VideoFrame.from_image(Image.new("RGB", (640, 480), (4, 15, 19)))
        except (OSError, ValueError):
            frame = VideoFrame.from_image(Image.new("RGB", (640, 480), (4, 15, 19)))
        frame.pts = pts
        frame.time_base = time_base
        return frame


class RobotAudioTrack(AudioStreamTrack):
    """Pace the robot's existing 48 kHz PCM output into WebRTC audio frames."""

    def __init__(self):
        super().__init__()
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._pts = 0

    def push(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            self._buffer.extend(pcm)
            max_bytes = ROBOT_AUDIO_SAMPLE_RATE * 2
            if len(self._buffer) > max_bytes:
                del self._buffer[:-max_bytes]

    async def recv(self):
        await asyncio.sleep(ROBOT_AUDIO_FRAME_SAMPLES / ROBOT_AUDIO_SAMPLE_RATE)
        needed = ROBOT_AUDIO_FRAME_SAMPLES * 2
        with self._lock:
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
    ):
        self.node = node
        self.signaling_url = signaling_url
        self.robot_id = robot_id
        self.token = token
        self.servo_step_size = max(0.1, min(65535.0, float(servo_step_size)))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None
        self._controller_peer_id: str | None = None
        self._peer: RTCPeerConnection | None = None
        self._channel = None
        self._video_track: CameraVideoTrack | None = None
        self._audio_track: RobotAudioTrack | None = None
        self._camera_renew_task: asyncio.Task | None = None
        self._audio_tasks: set[asyncio.Task] = set()
        self._last_jpeg = b""
        self._frame_lock = threading.Lock()
        self._last_sequence = -1
        self._remote_voice_active = False
        self._camera_client_id = f"webrtc:{robot_id}"[:96]
        self._neck_kinematics = load_neck_kinematics()

        self._camera_command_pub = node.create_publisher(String, CAMERA_COMMAND_TOPIC, 10)
        self._motor_pub = node.create_publisher(String, MOTOR_REMOTE_TOPIC, 10)
        self._action_pub = node.create_publisher(String, ACTION_REQUEST_TOPIC, 10)
        self._remote_audio_pub = node.create_publisher(UInt8MultiArray, REMOTE_AUDIO_PCM_TOPIC, 10)
        self._voice_state_pub = node.create_publisher(String, REMOTE_VOICE_STATE_TOPIC, 10)
        self._camera_sub = node.create_subscription(
            CompressedImage,
            CAMERA_FRAME_TOPIC,
            self._on_camera_frame,
            qos_profile_sensor_data,
        )
        self._audio_sub = node.create_subscription(
            UInt8MultiArray,
            "audio_output",
            self._on_audio_output,
            10,
        )
        self._dialog_sub = node.create_subscription(
            String,
            "screen_dialog",
            self._on_dialog,
            10,
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
            loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._stop_robot_outputs()

    def latest_camera_frame(self) -> bytes:
        with self._frame_lock:
            return self._last_jpeg

    def _on_camera_frame(self, message: CompressedImage) -> None:
        jpeg = bytes(message.data or b"")
        if jpeg:
            with self._frame_lock:
                self._last_jpeg = jpeg

    def _on_audio_output(self, message: UInt8MultiArray) -> None:
        track = self._audio_track
        if track is not None:
            track.push(bytes(message.data or b""))

    def _on_dialog(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            payload = {"ai_text": str(message.data or "")}
        text = payload.get("ai_text") if isinstance(payload, dict) else ""
        if isinstance(text, str) and text.strip():
            self._send_event_threadsafe({
                "type": "transcript",
                "speaker": "robot",
                "text": text.strip()[:1000],
            })

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
        while not self._stopping.is_set():
            try:
                async with websockets.connect(
                    self.signaling_url,
                    max_size=128 * 1024,
                    ping_interval=20,
                ) as socket:
                    self._socket = socket
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
                    self.node.get_logger().warning(f"信令连接断开，将重试: {exc}")
            finally:
                await self._close_peer()
                self._socket = None
            if not self._stopping.is_set():
                await asyncio.sleep(SIGNALING_RECONNECT_DELAY_SEC)

    async def _handle_signaling(self, raw) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return
        if message.get("type") == "signal":
            if isinstance(message.get("from"), str):
                self._controller_peer_id = message["from"]
            signal = message.get("signal")
            if isinstance(signal, dict):
                await self._handle_signal(signal)
        elif message.get("type") == "peer-left":
            await self._close_peer()
        elif message.get("type") == "peer-ready" and message.get("role") == "controller":
            self._controller_peer_id = str(message.get("peerId") or "") or None

    async def _handle_signal(self, signal: dict) -> None:
        kind = signal.get("type")
        if kind == "offer" and isinstance(signal.get("sdp"), str):
            await self._accept_offer(signal["sdp"])
        elif kind == "candidate":
            await self._accept_candidate(signal)

    async def _accept_offer(self, sdp: str) -> None:
        controller_peer_id = self._controller_peer_id
        await self._close_peer()
        # _close_peer() clears the old session identity. Preserve the sender
        # of this offer so the answer can be routed back through signaling.
        self._controller_peer_id = controller_peer_id
        peer = RTCPeerConnection()
        self._peer = peer
        self._video_track = CameraVideoTrack(self)
        self._audio_track = RobotAudioTrack()
        peer.addTrack(self._video_track)
        peer.addTrack(self._audio_track)
        self._acquire_camera()

        @peer.on("datachannel")
        def on_datachannel(channel):
            self._channel = channel

            @channel.on("open")
            def on_open():
                self._send_event({
                    "type": "status",
                    "value": "机器人 WebRTC 网关已就绪",
                })

            @channel.on("message")
            def on_message(message):
                self._on_data_message(message)

            @channel.on("close")
            def on_close():
                self._stop_robot_outputs()

        @peer.on("track")
        def on_track(track):
            if track.kind == "audio":
                task = asyncio.create_task(self._consume_remote_audio(track))
                self._audio_tasks.add(task)
                task.add_done_callback(self._audio_tasks.discard)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange():
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await self._close_peer()

        await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)
        await self._wait_ice_complete(peer)
        local = peer.localDescription
        if local and self._socket is not None and self._controller_peer_id:
            await self._socket.send(json.dumps({
                "type": "signal",
                "to": self._controller_peer_id,
                "signal": {"type": "answer", "sdp": local.sdp},
            }, separators=(",", ":")))

    async def _wait_ice_complete(self, peer: RTCPeerConnection) -> None:
        deadline = time.monotonic() + 8.0
        while peer.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def _accept_candidate(self, signal: dict) -> None:
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

    def _on_data_message(self, raw) -> None:
        message = decode_remote_message(raw)
        if message is None:
            return
        sequence = int(message["seq"])
        if sequence <= self._last_sequence:
            return
        self._last_sequence = sequence
        message_type = message["type"]
        if message_type == "control":
            self._publish_control(message["vector"])
        elif message_type == "stop":
            self._stop_robot_outputs()
        elif message_type == "action":
            self._publish_action(message["name"])
        elif message_type == "voice":
            self._set_remote_voice(message["state"] == "start")

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
        self._action_pub.publish(String(data=json.dumps({
            "name": name,
            "arguments": {},
            "request_id": new_action_request_id(),
            "source": REMOTE_SOURCE,
        }, ensure_ascii=False, separators=(",", ":"))))

    def _stop_robot_outputs(self) -> None:
        self._motor_pub.publish(String(data=json.dumps(STOP_COMMAND, separators=(",", ":"))))
        self._action_pub.publish(String(data=json.dumps({
            "name": "stop_all",
            "arguments": {},
            "request_id": new_action_request_id(),
            "source": REMOTE_SAFETY_SOURCE,
        }, separators=(",", ":"))))
        if self._remote_voice_active:
            self._set_remote_voice(False)
        self._release_camera()

    def _set_remote_voice(self, active: bool) -> None:
        if active == self._remote_voice_active:
            return
        self._remote_voice_active = active
        self._voice_state_pub.publish(String(data=encode_voice_state("start" if active else "stop")))

    async def _consume_remote_audio(self, track) -> None:
        while not self._stopping.is_set() and track.readyState == "live":
            try:
                frame = await track.recv()
            except Exception:
                return
            if not self._remote_voice_active:
                continue
            try:
                samples = frame.to_ndarray(format="s16")
                if samples.ndim > 1:
                    samples = np.mean(samples, axis=0)
                samples = _resample_mono(
                    samples,
                    int(frame.sample_rate or 48_000),
                    REMOTE_AUDIO_SAMPLE_RATE,
                )
                message = UInt8MultiArray()
                message.data = list(samples.astype(np.int16).tobytes())
                self._remote_audio_pub.publish(message)
            except (AttributeError, TypeError, ValueError):
                continue

    def _send_event_threadsafe(self, event: dict) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._send_event, event)

    def _send_event(self, event: dict) -> None:
        channel = self._channel
        if channel is None or channel.readyState != "open":
            return
        channel.send(json.dumps({"type": "event", "event": event}, ensure_ascii=False, separators=(",", ":")))

    def _acquire_camera(self) -> None:
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
            task.cancel()
        self._camera_command_pub.publish(String(data=encode_camera_command(
            "release", self._camera_client_id
        )))

    async def _close_peer(self) -> None:
        for task in list(self._audio_tasks):
            task.cancel()
        self._audio_tasks.clear()
        peer = self._peer
        self._peer = None
        self._channel = None
        self._controller_peer_id = None
        self._video_track = None
        self._audio_track = None
        self._last_sequence = -1
        self._stop_robot_outputs()
        if peer is not None:
            await peer.close()


class RemoteWebRtcNode(Node):
    def __init__(self):
        super().__init__("remote_webrtc_gateway_node")
        config = _remote_config()
        signaling_url = os.environ.get(
            "WALLE_WEBRTC_SIGNALING_URL",
            str(config.get("signaling_url", "ws://127.0.0.1:8787/signal")),
        ).strip()
        robot_id = os.environ.get("WALLE_WEBRTC_ROBOT_ID", str(config.get("robot_id", "WALLY-01"))).strip()
        token = os.environ.get("WALLE_WEBRTC_SIGNALING_TOKEN", str(config.get("token", "")))
        step_size = config.get("servo_step_size", 50.0)
        self._gateway = RemoteWebRtcGateway(
            self,
            signaling_url=signaling_url,
            robot_id=robot_id,
            token=token,
            servo_step_size=step_size,
        )
        self._gateway.start()
        self.get_logger().info(
            f"WebRTC 机器人网关上线: room={robot_id}, signaling={signaling_url}"
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
