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
    def test_listening_and_model_expression_publish_low_priority_targets(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]

        interval, timer = node.timers[0]
        self.assertEqual(interval, 2.0)
        node.subscriptions["dialog_motion_vad"](_String("speech_started"))
        timer()
        node.subscriptions["dialog_expression"](_String(json.dumps({
            "expression": "surprised", "intensity": "high", "turn_id": "t1"
        })))
        node.subscriptions["tts_text"](_String("你好，我在。"))
        timer()

        self.assertEqual(len(publisher.messages), 3)
        listening, listening_refresh, speaking = [
            json.loads(message.data) for message in publisher.messages
        ]
        self.assertEqual(listening["source"], "dialog_motion")
        self.assertEqual(listening["step_size"], 12.0)
        self.assertEqual(listening_refresh["source"], "dialog_motion")
        self.assertEqual(speaking["source"], "dialog_motion")
        self.assertEqual(
            speaking["targets"]["neck_top"],
            max(node._servos["neck_top"]["limit_1"], node._servos["neck_top"]["limit_2"]),
        )
        self.assertEqual(
            speaking["targets"]["neck_bottom"],
            max(node._servos["neck_bottom"]["limit_1"], node._servos["neck_bottom"]["limit_2"]),
        )

        for payload in (listening, speaking):
            targets = payload["targets"]
            for name, target in targets.items():
                servo = node._servos[name]
                self.assertGreaterEqual(target, min(servo["limit_1"], servo["limit_2"]))
                self.assertLessEqual(target, max(servo["limit_1"], servo["limit_2"]))

    def test_motion_is_quiet_before_wake_vad_and_after_speech_ends(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/servo_targets/dialog_expression"]
        _, timer = node.timers[0]

        timer()
        node.subscriptions["dialog_motion_vad"](_String("speech_started"))
        self.assertEqual(len(publisher.messages), 1)
        timer()
        node.subscriptions["dialog_motion_vad"](_String("speech_ended"))
        timer()

        self.assertEqual(len(publisher.messages), 2)

    def test_playback_completion_stops_speaking_motion_until_vad_detects_speech(self):
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
            for pose in (sampler.listening_pose(), sampler.speaking_pose()):
                self.assertEqual(pose["eye_l"] - pose["eye_r"], expected_eye_gap)
                for name, target in pose.items():
                    servo = servos[name]
                    self.assertGreaterEqual(target, min(servo["limit_1"], servo["limit_2"]))
                    self.assertLessEqual(target, max(servo["limit_1"], servo["limit_2"]))


if __name__ == "__main__":
    unittest.main()
