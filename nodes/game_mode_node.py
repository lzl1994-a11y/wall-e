#!/usr/bin/env python3
"""Coordinate the FC menu/session while existing nodes retain hardware ownership."""

import json
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

from services.fc_game_session import FcGameSession
from services.game_mode import GameModeController, InvalidGameTransition
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_REQUEST_TOPIC,
    GAME_MODE_STATE_TOPIC,
    decode_game_message,
    encode_game_frame,
)


ROOT = Path(__file__).resolve().parent.parent


class _RosPlaybackSink:
    def __init__(self, publisher) -> None:
        self._publisher = publisher
        self.muted = False

    def play(self, samples: np.ndarray) -> None:
        if self.muted or samples is None or len(samples) == 0:
            return
        self._publisher.publish(UInt8MultiArray(data=samples.astype(np.int16, copy=False).tobytes()))

    def mark_turn_end(self) -> None:
        self._publisher.publish(UInt8MultiArray(data=[]))


class GameModeNode(Node):
    """Own state, controller grab, ROM menu, and raw game outputs only."""

    def __init__(self):
        super().__init__("game_mode_node")
        self._controller = GameModeController()
        self._state_pub = self.create_publisher(String, GAME_MODE_STATE_TOPIC, 10)
        self._frame_pub = self.create_publisher(UInt8MultiArray, GAME_FRAME_TOPIC, 1)
        self._audio_pub = self.create_publisher(UInt8MultiArray, "audio_output", 10)
        self._playback = _RosPlaybackSink(self._audio_pub)
        self.create_subscription(String, GAME_MODE_REQUEST_TOPIC, self._on_request, 10)
        self.create_subscription(String, "llm_busy", self._on_llm_busy, 10)
        self._session: FcGameSession | None = None
        self._session_thread: threading.Thread | None = None
        self._session_stop = threading.Event()
        self._controller_path = "/dev/input/event2"
        self._state_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_sequence = 0
        self._published_frame_sequence = 0
        self._awaiting_audio_end = False
        self._audio_finished = False
        self._exit_timer: threading.Timer | None = None
        self._publish_state()
        self.create_timer(1.0, self._publish_state)
        self.create_timer(0.1, self._flush_game_frame)

    def _on_request(self, message):
        try:
            payload = decode_game_message(message.data)
            request = payload.get("request") if payload is not None else None
        except (TypeError, AttributeError):
            return
        with self._state_lock:
            try:
                if request == "toggle" and self._controller.mode.value == "robot":
                    controller_path = payload.get("controller")
                    if isinstance(controller_path, str) and controller_path:
                        self._controller_path = controller_path
                    self._controller.request_enter()
                    self._publish_state()
                elif request == "game_surface_ready":
                    self._controller.game_surface_ready()
                    self._publish_state()
                    self._start_session()
                elif request == "controller_disconnected":
                    if self._controller.mode.value == "entering":
                        self._controller.request_exit()
                        self._publish_state()
                        self._controller.robot_surface_ready()
                        self._publish_state()
                    else:
                        self._request_session_stop()
                elif request == "pause":
                    self._controller.pause_for_fault()
                    self._publish_state()
                elif request == "resume":
                    self._controller.resume_game()
                    self._publish_state()
                else:
                    return
            except InvalidGameTransition as exc:
                self.get_logger().warning(str(exc))

    def _start_session(self):
        if self._session_thread is not None and self._session_thread.is_alive():
            return
        self._session_stop.clear()
        self._session = FcGameSession(
            core_path="/root/libretro-fceumm/fceumm_libretro.so",
            rom_directory=ROOT / "data" / "games" / "roms",
            controller_path=self._controller_path,
            on_frame=self._queue_game_frame,
            playback=self._playback,
            gain=0.4,
            on_game_started=self._on_game_started,
            on_audio_end_queued=self._prepare_audio_end,
        )
        self._session_thread = threading.Thread(
            target=self._run_session, name="fc-game-session", daemon=True
        )
        self._session_thread.start()

    def _run_session(self):
        try:
            self._session.run(self._session_stop)
        except Exception as exc:
            self.get_logger().error(f"FC 游戏会话失败: {exc}")
        finally:
            self._finish_session()

    def _on_game_started(self, rom: Path):
        with self._state_lock:
            if self._controller.mode.value == "menu":
                self._controller.start_game()
                self.get_logger().info(f"启动 FC 游戏: {rom.name}")
                self._publish_state()

    def _request_session_stop(self):
        self._session_stop.set()

    def _finish_session(self):
        with self._state_lock:
            mode = self._controller.mode.value
            if mode in {"menu", "playing", "paused"}:
                self._controller.request_exit()
                self._publish_state()
            if self._controller.mode.value == "exiting":
                if self._audio_finished:
                    self._complete_exit()
                else:
                    self._start_exit_timeout()
        self._session = None

    def _prepare_audio_end(self):
        with self._state_lock:
            self._awaiting_audio_end = True
            self._audio_finished = False

    def _start_exit_timeout(self):
        if self._exit_timer is not None:
            self._exit_timer.cancel()
        self._exit_timer = threading.Timer(15.0, self._complete_exit)
        self._exit_timer.daemon = True
        self._exit_timer.start()

    def _complete_exit(self):
        with self._state_lock:
            if self._controller.mode.value != "exiting":
                return
            timer = self._exit_timer
            self._exit_timer = None
            if timer is not None and timer is not threading.current_thread():
                timer.cancel()
            self._awaiting_audio_end = False
            self._audio_finished = False
            self._controller.robot_surface_ready()
            self._publish_state()

    def _queue_game_frame(self, raw: bytes, width: int, height: int, pitch: int):
        with self._frame_lock:
            self._latest_frame = (raw, width, height, pitch)
            self._frame_sequence += 1

    def _flush_game_frame(self):
        with self._frame_lock:
            if self._frame_sequence == self._published_frame_sequence:
                return
            frame = self._latest_frame
            self._published_frame_sequence = self._frame_sequence
        if frame is None:
            return
        raw, width, height, pitch = frame
        self._frame_pub.publish(UInt8MultiArray(
            data=encode_game_frame(raw, width, height, pitch)
        ))

    def _on_llm_busy(self, message):
        if message.data == "busy":
            self._playback.muted = True
        elif message.data == "idle":
            self._playback.muted = False
            with self._state_lock:
                if self._awaiting_audio_end:
                    self._audio_finished = True
                    if self._controller.mode.value == "exiting":
                        self._complete_exit()

    def _publish_state(self):
        policy = self._controller.policy
        self._state_pub.publish(String(data=json.dumps({
            "mode": self._controller.mode.value,
            "robot_input": policy.robot_input,
            "recording": policy.recording,
            "game_input": policy.game_input,
            "game_audio": policy.game_audio,
            "motors_must_stop": policy.motors_must_stop,
        }, separators=(",", ":"))))

    def destroy_node(self):
        self._request_session_stop()
        if self._exit_timer is not None:
            self._exit_timer.cancel()
        thread = self._session_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GameModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
