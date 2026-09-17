"""Unit tests for the pure LLM tool proposal evaluation service."""

import json
import unittest

from services.llm_tool_proposal import (
    ROUTE_ACTION,
    ROUTE_CAMERA_INSPECTION,
    ROUTE_CAMERA_PHOTO,
    ROUTE_CONDITIONAL_TASK,
    REJECTION_MALFORMED_ARGUMENTS,
    REJECTION_POLICY,
    STREAM_CONTROL_BREAK,
    STREAM_CONTROL_CONTINUE,
    STREAM_CONTROL_PROCEED,
    LLMToolProposalEvaluator,
    ToolProposalDecision,
    evaluate_tool_proposal,
)


class LLMToolProposalTests(unittest.TestCase):
    def test_1_valid_ordinary_action_proposal(self):
        tool_call = {
            "name": "play_sequence",
            "arguments": json.dumps({"sequence_name": "turn_head_left"}),
        }
        decision = evaluate_tool_proposal(tool_call, user_prompt="向左转个头")
        self.assertTrue(decision.is_accepted)
        self.assertFalse(decision.is_rejected)
        self.assertEqual(decision.route, ROUTE_ACTION)
        self.assertTrue(decision.is_action)
        self.assertEqual(decision.action_name, "play_sequence")
        self.assertEqual(decision.arguments, {"sequence_name": "turn_head_left"})
        self.assertIsNone(decision.rejection_reason)
        self.assertEqual(decision.stream_control, STREAM_CONTROL_PROCEED)
        self.assertFalse(decision.should_break)
        self.assertFalse(decision.should_continue)

    def test_2_arguments_as_valid_json_object_and_dict(self):
        # 1. String JSON arguments
        tool_call_str = {
            "name": "express_emotion",
            "arguments": '{"emotion": "happy"}',
        }
        decision_str = evaluate_tool_proposal(tool_call_str, user_prompt="做个开心表情")
        self.assertTrue(decision_str.is_accepted)
        self.assertEqual(decision_str.arguments, {"emotion": "happy"})

        # 2. Already parsed dict arguments
        tool_call_dict = {
            "name": "express_emotion",
            "arguments": {"emotion": "happy"},
        }
        decision_dict = evaluate_tool_proposal(tool_call_dict, user_prompt="做个开心表情")
        self.assertTrue(decision_dict.is_accepted)
        self.assertEqual(decision_dict.arguments, {"emotion": "happy"})

    def test_3_malformed_json_rejects_with_invalid_arguments_and_breaks(self):
        tool_call = {
            "name": "play_sequence",
            "arguments": '{"sequence_name": ',
        }
        decision = evaluate_tool_proposal(tool_call, user_prompt="转头")
        self.assertFalse(decision.is_accepted)
        self.assertTrue(decision.is_rejected)
        self.assertIsNone(decision.route)
        self.assertEqual(decision.action_name, "play_sequence")
        self.assertEqual(decision.rejection_reason, "invalid_arguments")
        self.assertEqual(decision.rejection_kind, REJECTION_MALFORMED_ARGUMENTS)
        self.assertTrue(decision.is_malformed_arguments)
        self.assertEqual(decision.stream_control, STREAM_CONTROL_BREAK)
        self.assertTrue(decision.should_break)
        self.assertFalse(decision.should_continue)

    def test_4_json_decoded_non_object_safely_rejected(self):
        decoded_non_objects = [
            "123",
            '"just a string"',
            "[1, 2, 3]",
            "true",
        ]
        for val in decoded_non_objects:
            tool_call = {
                "name": "play_sequence",
                "arguments": val,
            }
            decision = evaluate_tool_proposal(tool_call, user_prompt="转头")
            self.assertFalse(decision.is_accepted, f"Failed for {val!r}")
            self.assertTrue(decision.is_rejected)
            self.assertEqual(decision.rejection_reason, "arguments_not_object")
            self.assertEqual(decision.rejection_kind, REJECTION_POLICY)
            self.assertFalse(decision.is_malformed_arguments)
            self.assertEqual(decision.stream_control, STREAM_CONTROL_BREAK)
            self.assertTrue(decision.should_break)

        for val in [123, True, ["list"]]:
            decision = evaluate_tool_proposal(
                {"name": "play_sequence", "arguments": val},
                user_prompt="转头",
            )
            self.assertEqual(decision.rejection_reason, "invalid_arguments")
            self.assertEqual(decision.rejection_kind, REJECTION_MALFORMED_ARGUMENTS)
            self.assertTrue(decision.is_malformed_arguments)

    def test_5_validate_action_call_rejection_and_reason_preserved(self):
        # Wording that explicitly negates action execution
        tool_call = {
            "name": "play_sequence",
            "arguments": json.dumps({"sequence_name": "turn_head_left"}),
        }
        decision = evaluate_tool_proposal(tool_call, user_prompt="不要转头")
        self.assertFalse(decision.is_accepted)
        self.assertTrue(decision.is_rejected)
        self.assertEqual(decision.rejection_reason, "negated_action")
        self.assertEqual(decision.rejection_kind, REJECTION_POLICY)
        self.assertFalse(decision.is_malformed_arguments)
        self.assertEqual(decision.stream_control, STREAM_CONTROL_BREAK)
        self.assertTrue(decision.should_break)
        self.assertFalse(decision.should_continue)
        self.assertIsNone(decision.route)

    def test_6_inspect_camera_normal_inspection_route(self):
        tool_call = {
            "name": "inspect_camera",
            "arguments": json.dumps({"question": "前面是什么"}),
        }
        decision = evaluate_tool_proposal(tool_call, user_prompt="看看前面是什么")
        self.assertTrue(decision.is_accepted)
        self.assertEqual(decision.route, ROUTE_CAMERA_INSPECTION)
        self.assertTrue(decision.is_camera_inspection)
        self.assertFalse(decision.is_camera_photo)
        self.assertEqual(decision.action_name, "inspect_camera")
        self.assertEqual(decision.arguments, {"question": "前面是什么"})
        self.assertEqual(decision.stream_control, STREAM_CONTROL_PROCEED)

    def test_7_inspect_camera_save_photo_route(self):
        tool_call = {
            "name": "inspect_camera",
            "arguments": json.dumps({"question": "前面是什么", "save_photo": True}),
        }
        decision = evaluate_tool_proposal(tool_call, user_prompt="拍照看看前面是什么")
        self.assertTrue(decision.is_accepted)
        self.assertEqual(decision.route, ROUTE_CAMERA_PHOTO)
        self.assertTrue(decision.is_camera_photo)
        self.assertFalse(decision.is_camera_inspection)
        self.assertEqual(decision.action_name, "inspect_camera")
        self.assertEqual(decision.arguments["save_photo"], True)
        self.assertEqual(decision.stream_control, STREAM_CONTROL_PROCEED)

    def test_8_run_conditional_task_route_with_plan(self):
        plan = {
            "observation": "人脸",
            "condition": "有人",
            "action_name": "play_sequence",
            "action_arguments": {"sequence_name": "basic_nod"},
        }
        tool_call = {
            "name": "run_conditional_task",
            "arguments": json.dumps(plan),
        }
        decision = evaluate_tool_proposal(
            tool_call,
            user_prompt="如果有人你就点头",
            conditional_request=True,
        )
        self.assertTrue(decision.is_accepted)
        self.assertEqual(decision.route, ROUTE_CONDITIONAL_TASK)
        self.assertTrue(decision.is_conditional_task)
        self.assertEqual(decision.action_name, "run_conditional_task")
        self.assertEqual(decision.plan, plan)
        self.assertEqual(decision.arguments, plan)
        self.assertEqual(decision.stream_control, STREAM_CONTROL_PROCEED)

    def test_9_conditional_task_split_action_rejected_and_continues(self):
        tool_call = {
            "name": "play_sequence",
            "arguments": json.dumps({"sequence_name": "turn_head_left"}),
        }
        decision = evaluate_tool_proposal(
            tool_call,
            user_prompt="如果有人你就点头",
            conditional_request=True,
        )
        self.assertFalse(decision.is_accepted)
        self.assertTrue(decision.is_rejected)
        self.assertEqual(decision.rejection_reason, "compound_task_must_stay_atomic")
        self.assertEqual(decision.stream_control, STREAM_CONTROL_CONTINUE)
        self.assertTrue(decision.should_continue)
        self.assertFalse(decision.should_break)
        self.assertIsNone(decision.route)

    def test_10_service_has_zero_side_effects(self):
        # 1. Inspect module imports: no rclpy, no ROS messages
        import services.llm_tool_proposal as module
        with open(module.__file__, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("rclpy", content)
        self.assertNotIn("std_msgs", content)
        self.assertNotIn("sensor_msgs", content)

        # 2. Immutability of decision object
        decision = evaluate_tool_proposal(
            {"name": "play_sequence", "arguments": '{"sequence_name": "basic_nod"}'},
            user_prompt="点头",
        )
        with self.assertRaises(Exception):
            decision.route = "camera_photo"  # FrozenInstanceError

        # 3. None or invalid tool_call handled safely without side effects
        bad_decision = evaluate_tool_proposal(None, user_prompt="你好")
        self.assertTrue(bad_decision.is_rejected)
        self.assertEqual(bad_decision.rejection_reason, "invalid_arguments")
        self.assertTrue(bad_decision.should_break)


if __name__ == "__main__":
    unittest.main()
