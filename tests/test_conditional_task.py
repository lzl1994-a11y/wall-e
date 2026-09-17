import json
import unittest

from services.conditional_task import (
    ConditionalTaskOutcome,
    build_conditional_task_failure_outcome,
    build_conditional_task_outcome,
    conditional_task_tool_schema,
    is_conditional_task_request,
    normalize_conditional_task_plan,
    parse_conditional_decision,
)


class ConditionalTaskTests(unittest.TestCase):
    def test_detection_is_generic_and_does_not_depend_on_one_object(self):
        self.assertTrue(is_conditional_task_request("看看前面，如果有人挥手你就点头"))
        self.assertTrue(is_conditional_task_request("发现红色物体就举手"))
        self.assertTrue(is_conditional_task_request("如果桌上没有杯子，就做个开心表情"))
        self.assertTrue(is_conditional_task_request("前面没人的话你就举手"))
        self.assertFalse(is_conditional_task_request("看看我手里有什么"))
        self.assertFalse(is_conditional_task_request("如果下雨会怎么样"))

    def test_plan_accepts_arbitrary_visual_condition_with_registered_action(self):
        plan = normalize_conditional_task_plan({
            "observation": "观察桌面上的物体和颜色",
            "condition": "桌面上至少有两个红色圆形物体",
            "action_name": "play_sequence",
            "action_arguments": {"sequence_name": "basic_nod"},
        })
        self.assertEqual(plan["condition"], "桌面上至少有两个红色圆形物体")

    def test_plan_rejects_unknown_action(self):
        base = {
            "observation": "观察前方",
            "condition": "前方没有障碍物",
            "action_arguments": {},
        }
        for action in ("run_shell",):
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError, "conditional_action_not_allowed"
            ):
                normalize_conditional_task_plan({**base, "action_name": action})

    def test_plan_accepts_bounded_conditional_chassis_motion(self):
        plan = normalize_conditional_task_plan({
            "observation": "观察前方",
            "condition": "前方没有人",
            "action_name": "move_chassis",
            "action_arguments": {"direction": "backward", "duration": 1},
        })

        self.assertEqual(plan["action_arguments"]["direction"], "backward")

    def test_plan_accepts_conditional_music_control(self):
        plan = normalize_conditional_task_plan({
            "observation": "观察前方",
            "condition": "前方没有人",
            "action_name": "control_music",
            "action_arguments": {"action": "play", "track": ""},
        })

        self.assertEqual(plan["action_arguments"]["action"], "play")

    def test_decision_parser_has_closed_vocabulary_and_fails_closed(self):
        self.assertEqual(
            parse_conditional_decision(
                '```json\n{"decision":"yes","evidence":"目标可见"}\n```'
            ),
            {"decision": "yes", "evidence": "目标可见"},
        )
        self.assertEqual(
            parse_conditional_decision('{"decision":"maybe"}')["decision"],
            "uncertain",
        )
        self.assertEqual(
            parse_conditional_decision("yes")["decision"],
            "uncertain",
        )

    def test_tool_schema_exposes_exact_action_arguments(self):
        schema = conditional_task_tool_schema()
        action_arguments = schema["properties"]["action_arguments"]
        self.assertFalse(action_arguments["additionalProperties"])
        sequence_names = action_arguments["properties"]["sequence_name"]["enum"]
        self.assertIn("basic_nod", sequence_names)
        self.assertIn("raise_hand", sequence_names)
        self.assertIn("right_hand_up", sequence_names)
        self.assertIn("left_hand_up", sequence_names)
        self.assertEqual(
            action_arguments["properties"]["direction"]["enum"],
            ["forward", "backward", "spin", "left", "right"],
        )

    def test_outcome_normal_answer_and_no_error(self):
        """1. 正常 answer 和 error=None。"""
        result = {"answer": "我已经观察并执行了动作。"}
        plan = {"action_name": "basic_nod", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.answer, "我已经观察并执行了动作。")
        self.assertIsNone(outcome.error)
        self.assertFalse(outcome.has_error)
        self.assertEqual(outcome.actions, [])

    def test_outcome_missing_answer_uses_fallback(self):
        """2. result 缺少 answer 时使用“这次任务没有完成。”。"""
        result = {}
        plan = {"action_name": "basic_nod", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.answer, "这次任务没有完成。")

    def test_outcome_empty_or_none_answer_uses_fallback(self):
        """3. answer 为空字符串或 None 时使用兜底。"""
        plan = {"action_name": "basic_nod", "action_arguments": {}}
        self.assertEqual(
            build_conditional_task_outcome({"answer": ""}, plan).answer,
            "这次任务没有完成。",
        )
        self.assertEqual(
            build_conditional_task_outcome({"answer": None}, plan).answer,
            "这次任务没有完成。",
        )

    def test_outcome_preserves_error(self):
        """4. result 带 error 时原样保留。"""
        result = {"error": "camera_timeout", "answer": "拍照超时"}
        plan = {"action_name": "basic_nod", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.error, "camera_timeout")
        self.assertTrue(outcome.has_error)

    def test_outcome_action_result_dict_generates_one_action(self):
        """5. action_result 为 dict 时生成一个 action。"""
        result = {
            "action_result": {
                "action": "play_sequence",
                "status": "succeeded",
                "request_id": "req-123",
            }
        }
        plan = {
            "action_name": "play_sequence",
            "action_arguments": {"sequence_name": "basic_nod"},
        }
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(len(outcome.actions), 1)

    def test_outcome_action_name_prefers_action_result_action(self):
        """6. action 名优先使用 action_result.action。"""
        result = {"action_result": {"action": "override_action"}}
        plan = {"action_name": "fallback_action", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.actions[0]["name"], "override_action")

    def test_outcome_action_name_falls_back_to_plan_action_name(self):
        """7. action_result 没有 action 时回退 plan.action_name。"""
        result = {"action_result": {"status": "succeeded"}}
        plan = {"action_name": "fallback_action", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.actions[0]["name"], "fallback_action")

    def test_outcome_action_arguments_from_plan(self):
        """8. arguments 来自 plan.action_arguments。"""
        result = {
            "action_result": {
                "action": "play_sequence",
                "arguments": {"ignored": True},
            }
        }
        plan = {
            "action_name": "play_sequence",
            "action_arguments": {"sequence_name": "wave_hello"},
        }
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(
            outcome.actions[0]["arguments"],
            json.dumps({"sequence_name": "wave_hello"}, ensure_ascii=False),
        )

    def test_outcome_chinese_arguments_ensure_ascii_false(self):
        """9. 中文 arguments 使用 ensure_ascii=False。"""
        plan = {
            "action_name": "control_music",
            "action_arguments": {"track": "周杰伦 晴天"},
        }
        result = {"action_result": {"action": "control_music"}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertIn("周杰伦 晴天", outcome.actions[0]["arguments"])
        self.assertNotIn("\\u", outcome.actions[0]["arguments"])

    def test_outcome_action_status_default_failed(self):
        """10. status 缺失时默认为 failed。"""
        result = {"action_result": {"action": "move_chassis"}}
        plan = {"action_name": "move_chassis", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.actions[0]["status"], "failed")

    def test_outcome_action_request_id_default_empty_string(self):
        """11. request_id 缺失时默认为空字符串。"""
        result = {"action_result": {"action": "move_chassis"}}
        plan = {"action_name": "move_chassis", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(outcome.actions[0]["request_id"], "")

    def test_outcome_action_result_non_dict_actions_empty(self):
        """12. action_result 不是 dict 时 actions=[]。"""
        plan = {"action_name": "move_chassis", "action_arguments": {}}
        for non_dict in (None, "failed", 123, [], True, False):
            with self.subTest(non_dict=non_dict):
                outcome = build_conditional_task_outcome({"action_result": non_dict}, plan)
                self.assertEqual(outcome.actions, [])

    def test_outcome_action_fields_exact_no_reason(self):
        """13. action 对象不包含 reason 等新增字段。"""
        result = {
            "action_result": {
                "action": "stop_all",
                "status": "succeeded",
                "request_id": "req-1",
                "reason": "should_not_appear",
            }
        }
        plan = {"action_name": "stop_all", "action_arguments": {}}
        outcome = build_conditional_task_outcome(result, plan)
        self.assertEqual(
            set(outcome.actions[0].keys()),
            {"name", "arguments", "status", "request_id"},
        )
        self.assertNotIn("reason", outcome.actions[0])

    def test_failure_outcome_fixed_failure_answer(self):
        """14. 异常 outcome 使用固定失败文案。"""
        outcome = build_conditional_task_failure_outcome("some_error")
        self.assertEqual(
            outcome.answer,
            "这个任务计划没有通过检查，所以我没有执行动作。",
        )
        self.assertEqual(outcome.actions, [])

    def test_failure_outcome_preserves_error(self):
        """15. 异常 outcome 保留原始 error。"""
        outcome = build_conditional_task_failure_outcome("specific_failure_reason")
        self.assertEqual(outcome.error, "specific_failure_reason")
        self.assertTrue(outcome.has_error)

    def test_outcome_frozen_dataclass(self):
        """16. ConditionalTaskOutcome 为 frozen dataclass。"""
        from dataclasses import FrozenInstanceError

        outcome = ConditionalTaskOutcome(answer="测试")
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            outcome.answer = "新回答"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            outcome.error = "新错误"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            outcome.actions = []  # type: ignore[misc]

    def test_service_is_pure_and_does_not_import_rclpy(self):
        """17. services/conditional_task.py 不导入 rclpy、不发布消息、不执行动作。"""
        import inspect
        import services.conditional_task as mod

        self.assertNotIn("rclpy", mod.__dict__)
        source = inspect.getsource(mod)
        self.assertNotIn("import rclpy", source)
        self.assertNotIn("from rclpy", source)
        self.assertNotIn("publish(", source)
        self.assertNotIn("invoke(", source)


if __name__ == "__main__":
    unittest.main()
