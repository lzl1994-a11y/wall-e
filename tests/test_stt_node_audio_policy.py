import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

class STTNodeAudioPolicyTests(unittest.TestCase):
    @staticmethod
    def _node():
        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy_node = types.ModuleType("rclpy.node")
        fake_rclpy_node.Node = object

        class String:
            def __init__(self, data=""):
                self.data = data

        fake_std_msgs = types.ModuleType("std_msgs")
        fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
        fake_std_msgs_msg.String = String
        modules = {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
        }
        sys.modules.pop("nodes.stt_ros_node", None)
        with patch.dict(sys.modules, modules):
            node_class = importlib.import_module("nodes.stt_ros_node").STTNode

        node = node_class.__new__(node_class)
        node._game_active = False
        node._llm_busy = False
        node._recording_paused = False
        node.stt_engine = MagicMock()
        node.get_logger = lambda: MagicMock()
        return node, String

    def test_dialog_and_game_pause_wake_inference_until_all_blockers_clear(self):
        node, message = self._node()

        node._on_llm_busy(message("busy"))
        node.stt_engine.pause.assert_called_once_with()

        node._on_game_state(message('{"mode":"playing"}'))
        node._on_llm_busy(message("idle"))
        node.stt_engine.resume.assert_not_called()

        node._on_game_state(message('{"mode":"robot"}'))
        node.stt_engine.resume.assert_called_once_with()

    def test_music_state_is_not_an_audio_capture_blocker(self):
        node, _message = self._node()

        self.assertFalse(hasattr(node, "_music_active"))
        node._sync_recording()

        node.stt_engine.pause.assert_not_called()


if __name__ == "__main__":
    unittest.main()
