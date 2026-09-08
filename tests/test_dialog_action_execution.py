"""Exercise real dialog dispatch with correlated executor acknowledgments, no ROS hardware."""

import importlib
import json
import sys
import threading
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from services.action_execution import CorrelatedActionExecutor
from services.action_status import build_action_status
from services.voice_chat_service import VoiceChatService


class DialogActionExecutionTests(unittest.TestCase):
    def _node(self, mode):
        class String:
            def __init__(self, data=""):
                self.data = data

        rclpy_node = types.ModuleType("rclpy.node")
        rclpy_node.Node = object
        std_msgs = types.ModuleType("std_msgs.msg")
        std_msgs.String = std_msgs.UInt8MultiArray = String
        modules = {
            "rclpy": types.ModuleType("rclpy"), "rclpy.node": rclpy_node,
            "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": std_msgs,
        }
        module_name = f"nodes.{'voice_chat_ros_node' if mode == 'voice' else 'llm_ros_node'}"
        sys.modules.pop(module_name, None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module(module_name)
        self.addCleanup(sys.modules.pop, module_name, None)
        node_class = module.VoiceChatNode if mode == "voice" else module.LLMBrainNode
        node = node_class.__new__(node_class)
        node._action_executor = CorrelatedActionExecutor()
        node.get_logger = lambda: MagicMock()
        publisher = MagicMock()
        publisher.get_subscription_count.return_value = 1
        node.action_pub = node.action_publisher = publisher
        node._game_mode = "robot"
        node._worker_running = True
        calls = [
            {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}},
            {"name": "play_sequence", "arguments": {"sequence_name": "basic_nod"}},
        ]
        if mode == "voice":
            service = VoiceChatService.__new__(VoiceChatService)
            service._chat_history = deque(maxlen=40)
            service._cancel_llm = threading.Event()
            service._last_llm_activity = 0
            service.multimodal = MagicMock()
            service.multimodal.build_audio_message.return_value = {"role": "user", "content": "audio"}
            service.system_prompt, service.model = "test", "test"
            service._stream_tool_calls = MagicMock(return_value=([
                {"name": "direct_answer", "arguments": {
                    "heard_text": "先挥手再点头", "response": "好的。",
                }}, *calls,
            ], ""))
            service.on_tool_call = node._on_tool_call
            service.on_llm_chunk = MagicMock()
            service.on_llm_reply = MagicMock()
            service._llm_done = MagicMock()
            node.vc = service
            run = lambda: service._send_to_llm("audio")
            reply = lambda: service.on_llm_reply.call_args.args[0]
        else:
            node.llm = MagicMock()
            node.llm.chat_stream.return_value = iter([
                {"type": "tool_call", "name": call["name"], "arguments": json.dumps(call["arguments"])}
                for call in calls
            ])
            node.chat_history = deque(maxlen=40)
            node.punctuations = {'。', '？', '.', '?', '！', '!'}
            for field in ("tts_publisher", "corrected_publisher", "full_ai_publisher",
                          "screen_dialog_publisher", "busy_publisher", "dialog_expression_publisher"):
                setattr(node, field, MagicMock())
            run = lambda: node._process_voice_task("turn", "先挥手再点头")
            reply = lambda: node.full_ai_publisher.publish.call_args.args[0].data
        return node, publisher, run, reply

    def _exercise(self, mode, first_status):
        node, publisher, run, reply = self._node(mode)
        commands = []
        first_published = threading.Event()

        def status(command, value):
            node._action_executor.accept_status(build_action_status(
                command["request_id"], command["name"], value, source="test_executor"
            ))

        def publish(message):
            command = json.loads(message.data)
            commands.append(command)
            status(command, "accepted")
            if len(commands) == 1:
                first_published.set()
            else:
                status(command, "completed")

        publisher.publish.side_effect = publish
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.assertTrue(first_published.wait(2))
            worker.join(0.05)
            self.assertTrue(worker.is_alive(), "accepted alone must not finish a turn")
            self.assertEqual(len(commands), 1, "the second action must wait for completion")
        finally:
            if commands:
                status(commands[0], first_status)
            worker.join(2)
        self.assertFalse(worker.is_alive())
        expected_count = 2 if first_status == "completed" else 1
        self.assertEqual(len(commands), expected_count)
        if first_status != "completed":
            self.assertIn("停止后续动作", reply())
        if mode == "text":
            results = [json.loads(m["content"])["status"] for m in node.chat_history if m["role"] == "tool"]
            self.assertEqual(results[0], first_status)

    def test_each_mode_waits_for_completion_before_publishing_next_action(self):
        for mode in ("voice", "text"):
            with self.subTest(mode=mode):
                self._exercise(mode, "completed")

    def test_failed_or_interrupted_action_aborts_remaining_actions_in_each_mode(self):
        for mode in ("voice", "text"):
            for status in ("failed", "interrupted", "rejected"):
                with self.subTest(mode=mode, status=status):
                    self._exercise(mode, status)


if __name__ == "__main__":
    unittest.main()
