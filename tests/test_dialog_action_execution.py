"""Exercise real dialog dispatch with correlated executor acknowledgments, no ROS hardware."""

import importlib
import json
import sys
import threading
import types
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from services.action.action_execution import CorrelatedActionExecutor
from services.action.action_status import build_action_status
from services.llm.voice_chat_service import VoiceChatService


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
            grounded_calls = [
                {
                    **call,
                    "arguments": {
                        **call["arguments"],
                        "grounding": "挥手" if index == 0 else "点头",
                    },
                }
                for index, call in enumerate(calls)
            ]
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
                    "intent_type": "execute_task",
                }}, *grounded_calls,
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

    def test_text_search_submits_visual_tree_and_reports_observation(self):
        node, publisher, _, reply = self._node("text")
        node.llm.chat_stream.return_value = iter([{
            "type": "tool_call", "name": "search_environment",
            "arguments": json.dumps({"target": "红色箱子"}),
        }])
        plans = []

        def execute(plan, **kwargs):
            plans.append(plan)
            return {"status": "success", "results": [{
                "action": "search_environment", "status": "found",
                "found": True, "response": "红色箱子在左前方。",
            }]}

        node._try_execute_native_plan = execute
        node._process_voice_task("turn", "这个好房间有一个红色的箱子里，找一下在哪里？")

        self.assertEqual(reply(), "红色箱子在左前方。")
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].root_type, "VisualSearch")
        self.assertEqual(plans[0].target, "红色箱子")
        call = next(m for m in node.chat_history if m.get("tool_calls"))["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"target": "红色箱子"})
        publisher.publish.assert_not_called()

    def test_text_search_reports_missing_native_owner_without_action_fallback(self):
        node, publisher, _, reply = self._node("text")
        node._try_execute_native_plan = MagicMock(return_value=None)
        node._process_visual_search("turn", "找红色箱子", {"target": "红色箱子"})

        self.assertIn("搜索行为树还没有就绪", reply())
        dialog = json.loads(node.screen_dialog_publisher.publish.call_args.args[0].data)
        self.assertEqual(dialog["error"], "native_behavior_tree_unavailable")
        publisher.publish.assert_not_called()

    def test_text_search_rejects_invalid_plan_without_submitting(self):
        node, publisher, _, _ = self._node("text")
        node._try_execute_native_plan = MagicMock()
        node._process_visual_search("turn", "找箱子", {"target": ""})
        node._try_execute_native_plan.assert_not_called()
        publisher.publish.assert_not_called()

    def test_text_search_view_returns_correlated_result_and_rejects_invalid_model_output(self):
        node, _, _, _ = self._node("text")
        node._visual_search_lock = threading.Lock()
        node.visual_search_status_publisher = MagicMock()
        node.tft_preview_settings = types.SimpleNamespace(recognition_duration_ms=100)
        node._run_camera_preview = MagicMock(return_value=types.SimpleNamespace(
            busy=False, last_frame=b"jpeg", error=None,
        ))
        request = {"request_id": "view-1", "plan_id": "search-1",
                   "target": "箱子", "question": "在哪里", "attempt": 1, "max_views": 3}
        for raw, expected in [
            (json.dumps({"status": "found", "evidence": "红色箱体", "response": "在左侧"}), "found"),
            ('{"status":"maybe"}', "uncertain"),
            ("invalid JSON", "uncertain"),
        ]:
            with self.subTest(raw=raw):
                node.llm.chat_stream.return_value = iter([{"type": "text", "content": raw}])
                node._serve_visual_search_request(request)
                result = json.loads(node.visual_search_status_publisher.publish.call_args.args[0].data)
                self.assertEqual(result["request_id"], "view-1")
                self.assertEqual(result["plan_id"], "search-1")
                self.assertEqual(result["status"], expected)
                kwargs = node.llm.chat_stream.call_args.kwargs
                self.assertFalse(kwargs["tools_enabled"])
                self.assertEqual(kwargs["image_base64"], "anBlZw==")

    def test_cancelled_search_does_not_publish_stale_final_reply(self):
        node, _, _, _ = self._node("text")
        node._voice_task_cancelled = lambda: True
        node._try_execute_native_plan = MagicMock(return_value=None)
        node._process_visual_search("turn", "找箱子", {"target": "箱子"})
        node.full_ai_publisher.publish.assert_not_called()

    def test_failed_or_interrupted_action_aborts_remaining_actions_in_each_mode(self):
        for mode in ("voice", "text"):
            for status in ("failed", "interrupted", "rejected"):
                with self.subTest(mode=mode, status=status):
                    self._exercise(mode, status)


if __name__ == "__main__":
    unittest.main()
