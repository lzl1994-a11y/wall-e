#!/usr/bin/env python3
"""ROS adapter for local music decoding, audio output, and spectrum values."""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray

from services.action_command import parse_action_request
from services.action_status import ACTION_STATUS_TOPIC, build_action_status
from services.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active
from services.music_player import DEFAULT_MUSIC_DIRECTORY, MusicPlayer
from services.music_protocol import (
    MUSIC_AUDIO_TOPIC,
    MUSIC_SPECTRUM_TOPIC,
    MUSIC_STATE_TOPIC,
    encode_music_state,
)


class MusicPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("music_player_node")
        directory = os.environ.get("WALI_MUSIC_DIR", str(DEFAULT_MUSIC_DIRECTORY))
        self._audio_pub = self.create_publisher(UInt8MultiArray, MUSIC_AUDIO_TOPIC, 10)
        self._spectrum_pub = self.create_publisher(
            Float32MultiArray, MUSIC_SPECTRUM_TOPIC, 1
        )
        self._state_pub = self.create_publisher(String, MUSIC_STATE_TOPIC, 10)
        self._status_pub = self.create_publisher(String, ACTION_STATUS_TOPIC, 10)
        self.create_subscription(String, "/action_cmd", self._on_action, 10)
        self.create_subscription(String, "llm_busy", self._on_dialog_state, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self._player = MusicPlayer(
            directory=directory,
            on_audio=lambda samples: self._audio_pub.publish(
                UInt8MultiArray(data=samples.tobytes())
            ),
            on_audio_end=lambda: self._audio_pub.publish(UInt8MultiArray(data=[])),
            on_spectrum=lambda levels: self._spectrum_pub.publish(
                Float32MultiArray(data=levels)
            ),
            on_state=self._publish_state,
        )
        self._publish_state("stopped", "", "")
        self.get_logger().info(f"本地音乐服务上线: {directory}")

    def _on_action(self, message) -> None:
        request = parse_action_request(message.data)
        if request is None:
            return
        name = request["name"]
        if name == "stop_all":
            self._player.stop()
            return
        if name != "control_music":
            return
        action = request["arguments"].get("action")
        if action == "stop":
            self._player.stop()
            self._publish_status(request, "completed")
            return
        if action != "play":
            self._publish_status(request, "rejected", "invalid_music_action")
            return
        try:
            track = self._player.play(request["arguments"].get("track", ""))
        except (OSError, ValueError) as exc:
            self._publish_status(request, "rejected", str(exc))
            return
        self._publish_status(request, "accepted")
        self._publish_status(request, "completed", track.name)

    def _on_dialog_state(self, message) -> None:
        if message.data in {"busy", "idle"}:
            self._player.set_speech_busy(message.data == "busy")

    def _on_game_state(self, message) -> None:
        if game_is_active(message.data):
            self._player.stop()

    def _publish_state(self, state: str, track: str, error: str) -> None:
        self._state_pub.publish(String(data=encode_music_state(state, track, error)))
        if state == "error":
            self.get_logger().error(f"音乐播放失败: {error}")

    def _publish_status(self, request, status: str, detail: str = "") -> None:
        request_id = request.get("request_id")
        if request_id:
            self._status_pub.publish(String(data=build_action_status(
                request_id, request["name"], status, source="music_player", detail=detail
            )))

    def destroy_node(self):
        self._player.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MusicPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
