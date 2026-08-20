import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class KeyboardSTTNodeTests(unittest.TestCase):
    @staticmethod
    def _load_node_class():
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
        sys.modules.pop("nodes.keyboard_stt_node", None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module("nodes.keyboard_stt_node")
        return module.KeyboardSTTNode

    def test_publish_text_trims_input_and_ignores_blank_lines(self):
        node_class = self._load_node_class()
        node = node_class.__new__(node_class)
        node._publisher = MagicMock()
        node.get_logger = lambda: MagicMock()

        self.assertFalse(node._publish_text("   "))
        self.assertTrue(node._publish_text("  你好，瓦力  "))

        message = node._publisher.publish.call_args.args[0]
        self.assertEqual(message.data, "你好，瓦力")
        node._publisher.publish.assert_called_once()
        sys.modules.pop("nodes.keyboard_stt_node", None)


if __name__ == "__main__":
    unittest.main()
