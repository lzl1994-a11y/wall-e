import unittest

from services.conditional_task import (
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

    def test_plan_rejects_unknown_or_locomotion_action(self):
        base = {
            "observation": "观察前方",
            "condition": "前方没有障碍物",
            "action_arguments": {},
        }
        for action in ("run_shell", "move_chassis"):
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError, "conditional_action_not_allowed"
            ):
                normalize_conditional_task_plan({**base, "action_name": action})

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


if __name__ == "__main__":
    unittest.main()
