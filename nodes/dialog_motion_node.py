#!/usr/bin/env python3
"""Emit one safe conversational pose for listening and speaking transitions.

This node deliberately contains only dialogue-state handling and pose sampling.
It sends ordinary ``manual_servo`` targets to the existing action pipeline, so
``sequence_ros_node`` remains the sole owner of servo interpolation, limits,
and hardware output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from services.dialog_motion_protocol import (
    DIALOG_MOTION_VAD_TOPIC,
    VAD_SPEECH_ENDED,
    VAD_SPEECH_STARTED,
)
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TARGET_TOPIC,
    DIALOG_EXPRESSION_TOPIC,
    decode_dialog_expression,
)
from services.servo_motion_config import resolve_servo_target
from services.tts_protocol import decode_turn_end


CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
BUSY_TOPIC = "llm_busy"
TTS_TOPIC = "tts_text"
MOTION_INTERVAL_SECONDS = 2.0


class DialogPoseSampler:
    """Sample coupled face, head-yaw, and neck-pitch poses within limits."""

    _STEP_SIZE = 12.0
    _DIALOG_RANGE_FRACTION = 0.08

    def __init__(self, servos, rng=None):
        self._servos = servos
        self._rng = rng or random.Random()
        self._eye_range = self._coupled_eye_range()
        self._eyebrow_open_range = self._eyebrow_open_range()
        self._head_yaw_range = self._half_offset_range("head_yaw")
        self._neck_pitch_range = self._neck_pitch_range()

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

    def _half_offset_range(self, name):
        cfg = self._servos[name]
        return (
            int((min(cfg["limit_1"], cfg["limit_2"]) - cfg["init"])
                * self._DIALOG_RANGE_FRACTION),
            int((max(cfg["limit_1"], cfg["limit_2"]) - cfg["init"])
                * self._DIALOG_RANGE_FRACTION),
        )

    def _neck_pitch_range(self):
        """Pitch both neck servos together, preserving their safe directions."""
        top = self._servos["neck_top"]
        bottom = self._servos["neck_bottom"]
        safe_pitch = min(
            max(top["limit_1"], top["limit_2"]) - top["init"],
            bottom["init"] - min(bottom["limit_1"], bottom["limit_2"]),
        )
        return (0, int(safe_pitch * self._DIALOG_RANGE_FRACTION))

    def _pose(self, eyebrow_range):
        eye_range = self._eye_range
        eye_offset = self._rng.randint(*eye_range)
        eyebrow_offset = self._rng.randint(*eyebrow_range)
        head_yaw_offset = self._rng.randint(*self._head_yaw_range)
        neck_pitch = self._rng.randint(*self._neck_pitch_range)
        targets = {
            # Equal eye offsets preserve the installed eye-pair gap.
            "eye_r": self._servos["eye_r"]["init"] + eye_offset,
            "eye_l": self._servos["eye_l"]["init"] + eye_offset,
            # The eyebrow servos are mirrored, so opening is +/- PWM.
            "eyebrow_r": self._servos["eyebrow_r"]["init"] + eyebrow_offset,
            "eyebrow_l": self._servos["eyebrow_l"]["init"] - eyebrow_offset,
            "head_yaw": self._servos["head_yaw"]["init"] + head_yaw_offset,
            "neck_top": self._servos["neck_top"]["init"] + neck_pitch,
            "neck_bottom": self._servos["neck_bottom"]["init"] - neck_pitch,
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
    required = {
        "eye_r", "eye_l", "eyebrow_r", "eyebrow_l",
        "head_yaw", "neck_top", "neck_bottom",
    }
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
        self._servos = _load_dialog_servos()
        self._sampler = DialogPoseSampler(self._servos)
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        self._listening_mode = str(
            (config.get("dialog_motion") or {}).get("listening_mode", "micro_motion")
        )
        if self._listening_mode not in {"micro_motion", "random_expression"}:
            self._listening_mode = "micro_motion"
        sequence_path = CONFIG_PATH.with_name("sequences.yaml")
        sequence_data = yaml.safe_load(sequence_path.read_text(encoding="utf-8")) or {}
        self._expression_poses = {
            name.removeprefix("expression_"): value
            for name, value in (sequence_data.get("poses") or {}).items()
            if name.startswith("expression_") and isinstance(value, dict)
        }
        self._neutral_targets = self._resolved_expression_targets("neutral")
        self._listening_choices = (
            "neutral", "listening", "thinking", "confused"
        )
        # No motion until the VAD reports actual human speech after wake-up.
        self._state = "idle"
        self._target_pub = self.create_publisher(
            String, DIALOG_EXPRESSION_TARGET_TOPIC, 10
        )
        self.create_subscription(
            String, DIALOG_MOTION_VAD_TOPIC, self._on_vad_state, 10
        )
        self.create_subscription(String, BUSY_TOPIC, self._on_playback_state, 10)
        self.create_subscription(String, TTS_TOPIC, self._on_tts_text, 10)
        self.create_subscription(
            String, DIALOG_EXPRESSION_TOPIC, self._on_expression, 10
        )
        self.create_timer(MOTION_INTERVAL_SECONDS, self._on_motion_timer)
        self.get_logger().info("对话姿态节点上线：VAD 人声/说话期间每2秒更新姿态")

    def _publish_pose(self, state, targets, step_size=None):
        payload = json.dumps({
            "targets": targets,
            "step_size": float(step_size or self._sampler.step_size),
            "source": "dialog_motion",
        }, ensure_ascii=False, separators=(",", ":"))
        self._target_pub.publish(String(data=payload))
        self._state = state
        self.get_logger().info(f"对话姿态 -> {state}: {targets}")

    def _on_vad_state(self, message):
        if message.data == VAD_SPEECH_STARTED:
            self._state = "listening"
            # Short utterances may end before the next two-second timer tick;
            # move once immediately when VAD positively identifies speech.
            if self._listening_mode == "random_expression":
                choice = random.choice(self._listening_choices)
                self._publish_expression_pose(choice, "low", "listening")
            else:
                self._publish_pose("listening", self._sampler.listening_pose())
        elif message.data == VAD_SPEECH_ENDED and self._state == "listening":
            self._state = "idle"

    def _on_tts_text(self, message):
        text = (message.data or "").strip()
        if not text or decode_turn_end(text) is not None:
            return
        # Streaming TTS may deliver several text segments.  One pose per
        # speaking transition is intentional; the trajectory node smooths it.
        if self._state != "speaking":
            self._publish_expression_pose("neutral", "low", "speaking")

    def _on_expression(self, message):
        value = decode_dialog_expression(message.data)
        if value is None:
            return
        self._publish_expression_pose(
            value["expression"], value["intensity"], "speaking"
        )

    def _publish_expression_pose(self, expression, intensity, state):
        pose = self._expression_poses.get(expression) or self._expression_poses.get("neutral", {})
        targets = self._resolved_expression_targets(expression)
        factors = {"low": 0.6, "medium": 0.85, "high": 1.0}
        factor = factors.get(intensity, 0.6)
        if self._neutral_targets and expression != "neutral":
            targets = {
                name: int(round(
                    self._neutral_targets.get(name, target)
                    + (target - self._neutral_targets.get(name, target)) * factor
                ))
                for name, target in targets.items()
            }
        self._publish_pose(
            state,
            targets,
            step_size=pose.get("default_step", self._sampler.step_size),
        )

    def _resolved_expression_targets(self, expression):
        pose = self._expression_poses.get(expression) or self._expression_poses.get("neutral", {})
        targets = {}
        for name, raw_target in pose.get("targets", {}).items():
            servo = self._servos.get(name)
            target = resolve_servo_target(servo, raw_target) if servo else None
            if target is not None:
                targets[name] = target
        return targets

    def _on_playback_state(self, message):
        # This existing state is emitted only after the queued audio has
        # physically drained. It ends body motion; it does not imply VAD is
        # currently hearing a person, so it must not enter listening mode.
        if message.data == "idle" and self._state == "speaking":
            self._publish_expression_pose("neutral", "low", "idle")

    def _on_motion_timer(self):
        """Refresh a conversational pose every two seconds during active phases."""
        if self._state == "listening" and self._listening_mode == "micro_motion":
            self._publish_pose("listening", self._sampler.listening_pose())


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
