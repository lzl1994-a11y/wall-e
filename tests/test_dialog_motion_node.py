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
    def test_listening_and_speaking_refresh_coupled_manual_servo_targets(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/action_cmd"]

        interval, timer = node.timers[0]
        self.assertEqual(interval, 2.0)
        node.subscriptions["dialog_motion_vad"](_String("speech_started"))
        timer()
        node.subscriptions["tts_text"](_String("你好，我在。"))
        node.subscriptions["tts_text"](_String("第二个流式分段。"))
        timer()

        self.assertEqual(len(publisher.messages), 4)
        listening, listening_refresh, speaking, speaking_refresh = [
            json.loads(message.data) for message in publisher.messages
        ]
        self.assertEqual(listening["name"], "manual_servo")
        self.assertEqual(listening["source"], "dialog_motion")
        self.assertEqual(listening["arguments"]["step_size"], 35.0)
        self.assertEqual(listening_refresh["source"], "dialog_motion")
        self.assertEqual(speaking["source"], "dialog_motion")
        self.assertEqual(speaking_refresh["source"], "dialog_motion")

        for payload in (listening, speaking, speaking_refresh):
            targets = payload["arguments"]["targets"]
            self.assertGreaterEqual(targets["eye_r"], 2000)
            self.assertLessEqual(targets["eye_r"], 4300)
            self.assertGreaterEqual(targets["eye_l"], 5000)
            self.assertLessEqual(targets["eye_l"], 7500)
            self.assertEqual(targets["eye_l"] - targets["eye_r"], 3500)
            self.assertGreaterEqual(targets["eyebrow_r"], 1920)
            self.assertLessEqual(targets["eyebrow_r"], 4200)
            self.assertGreaterEqual(targets["eyebrow_l"], 5700)
            self.assertLessEqual(targets["eyebrow_l"], 8000)
            self.assertTrue(1920 <= targets["head_yaw"] <= 7600)
            self.assertTrue(5000 <= targets["neck_top"] <= 6000)
            self.assertTrue(2000 <= targets["neck_bottom"] <= 5500)

    def test_motion_is_quiet_before_wake_vad_and_after_speech_ends(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/action_cmd"]
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
        publisher = node.publishers["/action_cmd"]
        _, timer = node.timers[0]

        node.subscriptions["tts_text"](_String("我正在说话。"))
        node.subscriptions["llm_busy"](_String("idle"))
        timer()

        self.assertEqual(len(publisher.messages), 1)

    def test_turn_end_marker_does_not_start_a_speaking_pose(self):
        module = _load_module()
        node = module.DialogMotionNode()
        publisher = node.publishers["/action_cmd"]

        node.subscriptions["tts_text"](
            _String('{"_wali_tts_control":"turn_end","turn_id":"turn-1"}')
        )

        self.assertEqual(publisher.messages, [])

    def test_sampler_keeps_coupled_targets_inside_limits(self):
        module = _load_module()
        sampler = module.DialogPoseSampler(
            module._load_dialog_servos(), rng=random.Random(7)
        )
        self.assertEqual(sampler._eye_range, (-500, 500))
        self.assertEqual(sampler._eyebrow_open_range, (0, 1140))
        self.assertEqual(sampler._head_yaw_range, (-1540, 1300))
        self.assertEqual(sampler._neck_pitch_range, (0, 500))
        for _ in range(100):
            for pose in (sampler.listening_pose(), sampler.speaking_pose()):
                self.assertEqual(pose["eye_l"] - pose["eye_r"], 3500)
                self.assertTrue(2000 <= pose["eye_r"] <= 4300)
                self.assertTrue(5000 <= pose["eye_l"] <= 7500)
                self.assertTrue(1920 <= pose["eyebrow_r"] <= 4200)
                self.assertTrue(5700 <= pose["eyebrow_l"] <= 8000)
                self.assertTrue(1920 <= pose["head_yaw"] <= 7600)
                self.assertTrue(5000 <= pose["neck_top"] <= 6000)
                self.assertTrue(2000 <= pose["neck_bottom"] <= 5500)


if __name__ == "__main__":
    unittest.main()
