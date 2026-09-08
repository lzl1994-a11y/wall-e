import importlib
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from services.music_protocol import encode_music_state


class _String:
    def __init__(self, data=""):
        self.data = data


class SerialMusicDisplayTests(unittest.TestCase):
    def setUp(self):
        fake_node = types.ModuleType("rclpy.node")
        fake_node.Node = object
        fake_qos = types.ModuleType("rclpy.qos")
        fake_qos.QoSProfile = MagicMock()
        fake_qos.DurabilityPolicy = MagicMock()
        fake_qos.ReliabilityPolicy = MagicMock()
        fake_messages = types.ModuleType("std_msgs.msg")
        fake_messages.String = _String
        sys.modules.pop("nodes.serial_ros_node", None)
        with patch.dict(sys.modules, {
            "rclpy": types.ModuleType("rclpy"),
            "rclpy.node": fake_node,
            "rclpy.qos": fake_qos,
            "std_msgs": types.ModuleType("std_msgs"),
            "std_msgs.msg": fake_messages,
        }):
            node_class = importlib.import_module("nodes.serial_ros_node").SerialNode
        self.node = node_class.__new__(node_class)
        self.node._music_active = False
        self.node.bridge = MagicMock()
        self.node.get_logger = lambda: MagicMock()

    def test_music_keeps_spectrum_during_voice_reply(self):
        for state in ("loading", "playing"):
            with self.subTest(state=state):
                self.node.bridge.reset_mock()
                self.node._on_music_state(_String(encode_music_state(state, "track")))
                self.node.screen_dialog_callback(_String(json.dumps({
                    "corrected_text": "播放音乐", "ai_text": "开始播放音乐了",
                })))
                self.node.bridge.send_raw.assert_called_once_with(
                    "eyeaction:talk\n", wake_screen=False
                )

    def test_confirmed_play_hint_wins_before_music_state_callback(self):
        self.node.screen_dialog_callback(_String(json.dumps({
            "ai_text": "开始播放音乐了",
            "actions": [{
                "name": "control_music",
                "arguments": {"action": "play", "track": ""},
                "status": "completed",
            }],
        })))

        self.node.bridge.send_raw.assert_called_once_with(
            "eyeaction:talk\n", wake_screen=False
        )

    def test_confirmed_stop_hint_restores_chat_before_state_callback(self):
        self.node._music_active = True
        self.node.screen_dialog_callback(_String(json.dumps({
            "ai_text": "音乐已停止",
            "actions": [{
                "name": "control_music",
                "arguments": '{"action":"stop"}',
                "status": "completed",
            }],
        })))

        self.assertIn(("openchat:1\n",), [
            call.args for call in self.node.bridge.send_raw.call_args_list
        ])

    def test_music_preserves_eye_and_motor_commands_without_chat_wake(self):
        self.node._on_music_state(_String(encode_music_state("playing")))

        self.node.tft_cmd_callback(_String("eyeaction:happy\n"))
        self.node.pca9685_callback(_String("pca9685:1"))

        calls = self.node.bridge.send_raw.call_args_list
        self.assertEqual(calls[0].args, ("eyeaction:happy\n",))
        self.assertEqual(calls[0].kwargs, {"wake_screen": False})
        self.assertEqual(calls[1].args, ("pca9685:1\n",))
        self.assertEqual(calls[1].kwargs, {"block": False, "wake_screen": False})

    def test_stopping_music_restores_chat_navigation(self):
        self.node._on_music_state(_String(encode_music_state("playing")))
        self.node._on_music_state(_String(encode_music_state("stopped")))

        self.node.screen_dialog_callback(_String(json.dumps({"ai_text": "已停止"})))

        self.assertIn(("openchat:1\n",), [
            call.args for call in self.node.bridge.send_raw.call_args_list
        ])

    def test_legacy_dialog_does_not_replace_music(self):
        self.node._on_music_state(_String(encode_music_state("playing")))

        self.node.you_callback(_String("你好"))
        self.node.ai_callback(_String("你好"))

        self.node.bridge.send_raw.assert_called_once_with(
            "eyeaction:talk\n", wake_screen=False
        )


if __name__ == "__main__":
    unittest.main()
