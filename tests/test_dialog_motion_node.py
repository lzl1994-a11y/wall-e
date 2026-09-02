import importlib
import json
import random
import sys
import types
import unittest
from unittest.mock import patch


class _String:
    def __init__(self, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def info(self, _message):
        pass


class _Node:
    def __init__(self, _name):
        self.publishers = {}
        self.subscriptions = {}
        self.timers = []

    def create_publisher(self, _message_type, topic, _qos):
        publisher = _Publisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return object()

    def create_timer(self, interval, callback):
        self.timers.append((interval, callback))
        return object()

    def get_logger(self):
        return _Logger()


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _Node
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
    }
    sys.modules.pop("nodes.dialog_motion_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.dialog_motion_node")


class DialogMotionNodeTests(unittest.TestCase):
    def test_speaking_motion_publishes_low_priority_targets(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]

        interval, timer = node.timers[0]
        self.assertEqual(interval, 2.0)
        node.subscriptions["tts_text"](_String("你好，我在。"))
        timer()

        self.assertEqual(len(publisher.messages), 2)
        speaking, speaking_refresh = [
            json.loads(message.data) for message in publisher.messages
        ]
        self.assertEqual(speaking["source"], "dialog_motion")
        self.assertEqual(speaking["step_size"], 24.0)
        self.assertEqual(speaking_refresh["source"], "dialog_motion")
        self.assertIn("neck_top", speaking["targets"])
        self.assertIn("neck_bottom", speaking["targets"])

        for payload in (speaking, speaking_refresh):
            targets = payload["targets"]
            for name, target in targets.items():
                servo = node._servos[name]
                self.assertGreaterEqual(target, min(servo["limit_1"], servo["limit_2"]))
                self.assertLessEqual(target, max(servo["limit_1"], servo["limit_2"]))

    def test_user_speech_has_no_motion_subscription_or_output(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]
        _, timer = node.timers[0]

        timer()
        timer()

        self.assertNotIn("dialog_motion_vad", node.subscriptions)
        self.assertEqual(publisher.messages, [])

    def test_playback_completion_stops_speaking_motion(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]
        _, timer = node.timers[0]

        node.subscriptions["tts_text"](_String("我正在说话。"))
        node.subscriptions["llm_busy"](_String("idle"))
        timer()

        self.assertEqual(len(publisher.messages), 2)
        neutral = json.loads(publisher.messages[-1].data)["targets"]
        self.assertEqual(neutral["neck_top"], node._servos["neck_top"]["init"])
        self.assertEqual(neutral["neck_bottom"], node._servos["neck_bottom"]["init"])

    def test_neutral_model_reply_uses_visible_speaking_micro_motion(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]

        node.subscriptions["dialog_expression"](_String(json.dumps({
            "expression": "neutral", "intensity": "low", "turn_id": "t1"
        })))
        first = json.loads(publisher.messages[-1].data)
        self.assertEqual(first["step_size"], 24.0)
        self.assertGreater(
            first["targets"]["neck_bottom"],
            node._servos["neck_bottom"]["init"],
        )

        _, timer = node.timers[0]
        timer()
        self.assertEqual(len(publisher.messages), 2)

    def test_turn_end_marker_does_not_start_a_speaking_pose(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]

        node.subscriptions["tts_text"](
            _String('{"_wali_tts_control":"turn_end","turn_id":"turn-1"}')
        )

        self.assertEqual(publisher.messages, [])

    def test_sampler_keeps_coupled_targets_inside_limits(self):
        module = _load_module()
        sampler = module.DialogPoseSampler(
            module._load_dialog_servos(), rng=random.Random(7)
        )
        servos = module._load_dialog_servos()
        expected_eye_gap = servos["eye_l"]["init"] - servos["eye_r"]["init"]
        for _ in range(100):
            pose = sampler.speaking_pose()
            self.assertEqual(pose["eye_l"] - pose["eye_r"], expected_eye_gap)
            for name, target in pose.items():
                servo = servos[name]
                self.assertGreaterEqual(target, min(servo["limit_1"], servo["limit_2"]))
                self.assertLessEqual(target, max(servo["limit_1"], servo["limit_2"]))


if __name__ == "__main__":
    unittest.main()
