#!/usr/bin/env python3
"""Emit one safe conversational pose for listening and speaking transitions.

This node deliberately contains only dialogue-state handling and pose sampling.
It sends ordinary ``manual_servo`` targets to the existing action pipeline, so
``sequence_ros_node`` remains the sole owner of servo interpolation, limits,
and hardware output.
"""

from __future__ import annotations

import random
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from services.action_command import build_action_cmd
from services.tts_protocol import decode_turn_end


CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
ACTION_TOPIC = "/action_cmd"
BUSY_TOPIC = "llm_busy"
TTS_TOPIC = "tts_text"


class DialogPoseSampler:
    """Sample coupled eye/eyebrow poses within configuration-derived limits."""

    _STEP_SIZE = 12.0
    _DIALOG_RANGE_FRACTION = 0.5

    def __init__(self, servos, rng=None):
        self._servos = servos
        self._rng = rng or random.Random()
        self._eye_range = self._coupled_eye_range()
        self._eyebrow_open_range = self._eyebrow_open_range()

    @staticmethod
    def _clamp(value, cfg):
        return int(max(min(cfg["limit_1"], cfg["limit_2"]), min(
            max(cfg["limit_1"], cfg["limit_2"]), value
        )))

    def _coupled_eye_range(self):
        """Return half of the common safe range for equal eye offsets."""
        lower = max(
            min(self._servos[name]["limit_1"], self._servos[name]["limit_2"])
            - self._servos[name]["init"]
            for name in ("eye_r", "eye_l")
        )
        upper = min(
            max(self._servos[name]["limit_1"], self._servos[name]["limit_2"])
            - self._servos[name]["init"]
            for name in ("eye_r", "eye_l")
        )
        return (
            int(lower * self._DIALOG_RANGE_FRACTION),
            int(upper * self._DIALOG_RANGE_FRACTION),
        )

    def _eyebrow_open_range(self):
        """Return half of the common safe range for mirrored eyebrow opening."""
        right = self._servos["eyebrow_r"]
        left = self._servos["eyebrow_l"]
        right_max = max(right["limit_1"], right["limit_2"])
        left_min = min(left["limit_1"], left["limit_2"])
        safe_open = min(right_max - right["init"], left["init"] - left_min)
        return (0, int(safe_open * self._DIALOG_RANGE_FRACTION))

    def _pose(self, eyebrow_range):
        eye_range = self._eye_range
        eye_offset = self._rng.randint(*eye_range)
        eyebrow_offset = self._rng.randint(*eyebrow_range)
        targets = {
            # Equal eye offsets preserve the installed eye-pair gap.
            "eye_r": self._servos["eye_r"]["init"] + eye_offset,
            "eye_l": self._servos["eye_l"]["init"] + eye_offset,
            # The eyebrow servos are mirrored, so opening is +/- PWM.
            "eyebrow_r": self._servos["eyebrow_r"]["init"] + eyebrow_offset,
            "eyebrow_l": self._servos["eyebrow_l"]["init"] - eyebrow_offset,
        }
        return {
            name: self._clamp(value, self._servos[name])
            for name, value in targets.items()
        }

    def listening_pose(self):
        # Listening holds the brows visibly open while looking to one side.
        lower, upper = self._eyebrow_open_range
        return self._pose((int(upper * 0.35), upper))

    def speaking_pose(self):
        return self._pose(self._eyebrow_open_range)

    @property
    def step_size(self):
        return self._STEP_SIZE


def _load_dialog_servos(config_path=CONFIG_PATH):
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取舵机配置: {exc}") from exc

    servos = {
        item.get("name"): item
        for item in config.get("servos", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required = {"eye_r", "eye_l", "eyebrow_r", "eyebrow_l"}
    missing = required - servos.keys()
    if missing:
        raise RuntimeError(f"对话姿态缺少舵机配置: {', '.join(sorted(missing))}")
    for name in required:
        cfg = servos[name]
        if not all(key in cfg for key in ("init", "limit_1", "limit_2")):
            raise RuntimeError(f"对话姿态舵机配置不完整: {name}")
    return {name: servos[name] for name in required}


class DialogMotionNode(Node):
    def __init__(self):
        super().__init__("dialog_motion_node")
        self._sampler = DialogPoseSampler(_load_dialog_servos())
        self._state = "idle"
        self._action_pub = self.create_publisher(String, ACTION_TOPIC, 10)
        self.create_subscription(String, BUSY_TOPIC, self._on_dialog_busy, 10)
        self.create_subscription(String, TTS_TOPIC, self._on_tts_text, 10)
        self.get_logger().info("对话姿态节点上线：等待录音或播报状态")

    def _publish_pose(self, state, targets):
        payload = build_action_cmd(
            "manual_servo",
            {"targets": targets, "step_size": self._sampler.step_size},
            source="dialog_motion",
        )
        self._action_pub.publish(String(data=payload))
        self._state = state
        self.get_logger().info(f"对话姿态 -> {state}: {targets}")

    def _on_dialog_busy(self, message):
        # Existing audio playback publishes idle only after the previous spoken
        # turn has physically finished; the next phase is listening/recording.
        if message.data == "idle" and self._state != "listening":
            self._publish_pose("listening", self._sampler.listening_pose())

    def _on_tts_text(self, message):
        text = (message.data or "").strip()
        if not text or decode_turn_end(text) is not None:
            return
        # Streaming TTS may deliver several text segments.  One pose per
        # speaking transition is intentional; the trajectory node smooths it.
        if self._state != "speaking":
            self._publish_pose("speaking", self._sampler.speaking_pose())


def main(args=None):
    rclpy.init(args=args)
    node = DialogMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
