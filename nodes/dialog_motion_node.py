#!/usr/bin/env python3
"""Emit safe conversational poses while the robot is speaking.

This node deliberately contains only dialogue-state handling and pose sampling.
It sends ordinary ``manual_servo`` targets to the existing action pipeline, so
``sequence_ros_node`` remains the sole owner of servo interpolation, limits,
and hardware output.
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from services.dialog_expression_pose import (
    resolve_dialog_expression_pose,
    resolve_pose_targets,
)
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TARGET_TOPIC,
    DIALOG_EXPRESSION_TOPIC,
    decode_dialog_expression,
)
from services.dialog_pose_sampler import DialogPoseSampler
from services.tts_protocol import decode_turn_end


CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
BUSY_TOPIC = "llm_busy"
TTS_TOPIC = "tts_text"
MOTION_INTERVAL_SECONDS = 2.0


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
        sequence_path = CONFIG_PATH.with_name("sequences.yaml")
        sequence_data = yaml.safe_load(sequence_path.read_text(encoding="utf-8")) or {}
        self._expression_poses = {
            name.removeprefix("expression_"): value
            for name, value in (sequence_data.get("poses") or {}).items()
            if name.startswith("expression_") and isinstance(value, dict)
        }
        self._neutral_targets = resolve_pose_targets(
            self._expression_poses.get("neutral"), self._servos
        )
        self._active_expression = "neutral"
        self._state = "idle"
        self._target_pub = self.create_publisher(
            String, DIALOG_EXPRESSION_TARGET_TOPIC, 10
        )
        self.create_subscription(String, BUSY_TOPIC, self._on_playback_state, 10)
        self.create_subscription(String, TTS_TOPIC, self._on_tts_text, 10)
        self.create_subscription(
            String, DIALOG_EXPRESSION_TOPIC, self._on_expression, 10
        )
        self.create_timer(MOTION_INTERVAL_SECONDS, self._on_motion_timer)
        self.get_logger().info("对话姿态节点上线：仅在瓦力说话期间每2秒更新姿态")

    def _publish_pose(self, state, targets, step_size=None):
        payload = json.dumps({
            "targets": targets,
            "step_size": float(step_size or self._sampler.step_size),
            "source": "dialog_motion",
        }, ensure_ascii=False, separators=(",", ":"))
        self._target_pub.publish(String(data=payload))
        self._state = state
        self.get_logger().info(f"对话姿态 -> {state}: {targets}")

    def _on_tts_text(self, message):
        text = (message.data or "").strip()
        if not text or decode_turn_end(text) is not None:
            return
        # Streaming TTS may deliver several text segments.  One pose per
        # speaking transition is intentional; the trajectory node smooths it.
        if self._state != "speaking":
            self._active_expression = "neutral"
            self._publish_pose("speaking", self._sampler.speaking_pose())

    def _on_expression(self, message):
        value = decode_dialog_expression(message.data)
        if value is None:
            return
        self._active_expression = value["expression"]
        if self._active_expression == "neutral":
            self._publish_pose("speaking", self._sampler.speaking_pose())
        else:
            self._publish_expression_pose(
                self._active_expression, value["intensity"], "speaking"
            )

    def _publish_expression_pose(self, expression, intensity, state):
        decision = resolve_dialog_expression_pose(
            self._expression_poses,
            self._servos,
            expression,
            intensity=intensity,
            default_step=self._sampler.step_size,
            neutral_targets=self._neutral_targets,
        )
        self._publish_pose(
            state,
            decision.targets,
            step_size=decision.step_size,
        )

    def _on_playback_state(self, message):
        # This existing state is emitted only after the queued audio has
        # physically drained. It ends body motion; it does not imply VAD is
        # currently hearing a person, so it must not enter listening mode.
        if message.data == "idle" and self._state == "speaking":
            self._active_expression = "neutral"
            self._publish_expression_pose("neutral", "low", "idle")

    def _on_motion_timer(self):
        """Refresh a conversational pose while the robot is speaking."""
        if self._state == "speaking" and self._active_expression == "neutral":
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
