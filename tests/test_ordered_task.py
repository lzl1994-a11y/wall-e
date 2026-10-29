import copy
import unittest

from services.action.action_intent_guard import (
    is_non_execution_request, validate_action_call,
)
from services.orchestration.ordered_task import (
    is_ordered_task_request, normalize_ordered_task_plan,
    ordered_task_tool_schema, validate_ordered_task_plan,
)


def step(name, arguments, grounding):
    return {"name": name, "arguments": arguments, "grounding": grounding}


class OrderedTaskPlanTests(unittest.TestCase):
    def test_model_wide_evidence_is_bound_to_the_requested_original_clause(self):
        text = "不要低头，然后挥手，再看看前面有什么"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "wave_hello"}, "不要低头，然后挥手"),
            step("inspect_camera", {}, "看看前面有什么"),
        ]}
        normalized = validate_ordered_task_plan(text, plan)
        self.assertEqual(normalized["steps"][0]["grounding"], "挥手")
        text = "不要低头，直接看看前面有什么"
        plan = {"steps": [step("inspect_camera", {}, text)]}
        self.assertEqual(validate_ordered_task_plan(text, plan)["steps"][0]["grounding"],
                         "直接看看前面有什么")

    def test_negated_lists_and_postposed_cancellation_are_not_required_actions(self):
        for text in ("不要低头、挥手，先看看", "低头、挥手都不要，看看即可"):
            plan = {"steps": [step("inspect_camera", {}, "看看")]}
            self.assertEqual(validate_ordered_task_plan(text, plan), plan)
            for sequence, grounding in (("head_down", "低头"), ("wave_hello", "挥手")):
                forbidden = {"steps": [
                    step("play_sequence", {"sequence_name": sequence}, grounding),
                    step("inspect_camera", {}, "看看"),
                ]}
                with self.assertRaisesRegex(ValueError, "negated_action"):
                    validate_ordered_task_plan(text, forbidden)

    def test_future_no_repeat_constraint_does_not_cancel_an_earlier_action(self):
        text = "先低头，看看前面有什么，再回正，不要再低头"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("inspect_camera", {}, "看看前面有什么"),
            step("play_sequence", {"sequence_name": "look_center"}, "回正"),
        ]}
        self.assertEqual(validate_ordered_task_plan(text, plan), plan)

    def test_omitted_prohibition_cannot_be_hidden_with_unrelated_grounding(self):
        text = "不要前进，然后挥手再看看"
        plan = {"steps": [
            step("move_chassis", {"direction": "forward", "duration": 1}, "然后"),
            step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
            step("inspect_camera", {}, "看看"),
        ]}
        with self.assertRaisesRegex(ValueError, "negated_action"):
            validate_ordered_task_plan(text, plan)

    def test_no_photo_constraint_on_later_view_preserves_an_earlier_requested_photo(self):
        text = "先低头拍张照，再看看右边，不要拍照"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("inspect_camera", {"save_photo": True}, "拍张照"),
            step("inspect_camera", {}, "看看右边"),
        ]}
        self.assertEqual(validate_ordered_task_plan(text, plan), plan)
        plan["steps"][-1]["arguments"]["save_photo"] = True
        with self.assertRaisesRegex(ValueError, "photo_save_forbidden"):
            validate_ordered_task_plan(text, plan)

    def test_photo_filter_cannot_remove_a_prohibition_on_observation(self):
        plan = {"steps": [
            step("inspect_camera", {"question": "前面有什么"}, "观察"),
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
        ]}
        with self.assertRaises(ValueError):
            validate_ordered_task_plan("不要拍照或进行任何观察，先低头", plan)
        with self.assertRaises(ValueError):
            validate_ordered_task_plan("不要拍照，先低头", {"steps": [
                step("inspect_camera", {}, "不要拍照"),
                step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            ]})

    def test_arbitrary_conditions_cannot_be_flattened_or_weakened(self):
        for condition in ("当红盒子出现时就挥手", "看到红盒子就挥手"):
            text = f"先低头，再看看前面，{condition}"
            plan = {"steps": [
                step("play_sequence", {"sequence_name": "head_down"}, "低头"),
                step("inspect_camera", {}, "看看前面"),
                step("play_sequence", {"sequence_name": "wave_hello"}, condition),
            ]}
            with self.assertRaisesRegex(ValueError, "condition_not_supported"):
                validate_ordered_task_plan(text, plan)
        for condition in ("找到后且它没有破损再挥手", "如果没找到就挥手",
                          "如果找到而且它是空的就挥手"):
            plan = {"steps": [
                step("play_sequence", {"sequence_name": "head_down"}, "低头"),
                step("search_environment", {"target": "红盒子", "on_found_actions": [
                    {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}},
                ]}, f"找红盒子，{condition}"),
            ]}
            with self.assertRaises(ValueError):
                validate_ordered_task_plan(f"先低头，然后找红盒子，{condition}", plan)

    def test_complete_low_head_wave_inspection_and_save_plan(self):
        text = "先低头，挥手，然后再看看前面有什么，并拍照保存。"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
            step("inspect_camera", {"question": "看看前面有什么", "save_photo": True},
                 "看看前面有什么，并拍照保存"),
        ]}
        self.assertTrue(is_ordered_task_request(text))
        normalized = validate_ordered_task_plan(text, plan)
        self.assertEqual(normalized, plan)
        self.assertEqual(validate_action_call(text, "run_ordered_task", plan), (True, ""))

    def test_turn_then_search_with_found_only_completion(self):
        text = "先转身，再找红盒子，找到后挥手。"
        plan = {"steps": [
            step("move_chassis", {"direction": "spin", "duration": 1}, "转身"),
            step("search_environment", {"target": "红盒子", "on_found_actions": [
                {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}},
            ]}, "找红盒子，找到后挥手"),
        ]}
        self.assertTrue(is_ordered_task_request(text))
        normalized = validate_ordered_task_plan(text, plan)
        self.assertEqual(normalized["steps"][1]["arguments"]["max_views"], 3)
        self.assertEqual(normalized["steps"][1]["arguments"]["on_found_actions"],
                         plan["steps"][1]["arguments"]["on_found_actions"])

    def test_different_durations_belong_to_their_literal_clauses(self):
        text = "先前进一秒然后后退两秒再看看"
        plan = {"steps": [
            step("move_chassis", {"direction": "forward", "duration": 1}, "前进一秒"),
            step("move_chassis", {"direction": "backward", "duration": 2}, "后退两秒"),
            step("inspect_camera", {}, "看看"),
        ]}
        self.assertEqual(validate_ordered_task_plan(text, plan), plan)

    def test_hint_never_needs_to_synthesize_an_action(self):
        for text in ("低头看看有什么", "先低头然后再看看", "转身找红盒子",
                     "先低头，然后看看前面有什么，再告诉我",
                     "先抬起右手，然后看看前面有什么", "先把右手举起来，然后看看前面有什么"):
            self.assertTrue(is_ordered_task_request(text), text)
        self.assertFalse(is_ordered_task_request("看看前面有什么"))

    def test_text_transformations_cannot_authorize_motion(self):
        for text in ("请复述：低头然后看看", "把‘低头然后看看’翻译成英文。",
                     "请读一下：低头然后看看", "请说“低头然后看看”。"):
            self.assertTrue(is_non_execution_request(text), text)
            self.assertFalse(is_ordered_task_request(text), text)
            plan = {"steps": [
                step("play_sequence", {"sequence_name": "head_down"}, "低头"),
                step("inspect_camera", {}, "看看"),
            ]}
            with self.assertRaisesRegex(ValueError, "non_command"):
                validate_ordered_task_plan(text, plan)

    def test_observed_object_names_are_not_text_transform_commands(self):
        for text in ("低头看看翻译机", "先挥手，再看看朗读器屏幕"):
            self.assertFalse(is_non_execution_request(text), text)
            self.assertTrue(is_ordered_task_request(text), text)

    def test_negation_outside_grounding_still_blocks_actions(self):
        text = "不要低头然后看看"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("inspect_camera", {}, "看看"),
        ]}
        with self.assertRaisesRegex(ValueError, "negated_action|stop_command_mismatch"):
            validate_ordered_task_plan(text, plan)

    def test_missing_explicit_motion_photo_and_reordered_steps_are_rejected(self):
        text = "先低头，挥手，然后看看前面有什么，并拍照保存。"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
            step("inspect_camera", {"save_photo": True}, "看看前面有什么，并拍照保存"),
        ]}
        omitted = copy.deepcopy(plan)
        omitted["steps"].pop(0)
        with self.assertRaisesRegex(ValueError, "missing_explicit_motion"):
            validate_ordered_task_plan(text, omitted)
        omitted = copy.deepcopy(plan)
        omitted["steps"][-1]["arguments"]["save_photo"] = False
        with self.assertRaisesRegex(ValueError, "missing_photo_save"):
            validate_ordered_task_plan(text, omitted)
        reordered = copy.deepcopy(plan)
        reordered["steps"][:2] = reversed(reordered["steps"][:2])
        with self.assertRaisesRegex(ValueError, "grounding_order_invalid"):
            validate_ordered_task_plan(text, reordered)

    def test_observed_person_actions_are_not_robot_motion_instructions(self):
        for question in ("看看那个人有没有低头", "看看那个人有没有转身", "看看那个人有没有拍照"):
            text = f"先挥手，然后{question}。"
            plan = {"steps": [
                step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
                step("inspect_camera", {"question": question}, question),
            ]}
            self.assertEqual(validate_ordered_task_plan(text, plan), plan)

    def test_pose_described_in_search_target_is_not_a_motion_instruction(self):
        text = "先挥手，然后找那个低头的人"
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
            step("search_environment", {"target": "低头的人"}, "找那个低头的人"),
        ]}
        self.assertEqual(validate_ordered_task_plan(text, plan)["steps"][1]["arguments"]["target"],
                         "低头的人")

    def test_found_only_action_cannot_be_flattened_into_unconditional_step(self):
        text = "转身找红盒子，找到后挥手"
        plan = {"steps": [
            step("move_chassis", {"direction": "spin", "duration": 1}, "转身"),
            step("search_environment", {"target": "红盒子"}, "找红盒子"),
            step("play_sequence", {"sequence_name": "wave_hello"}, "挥手"),
        ]}
        with self.assertRaisesRegex(ValueError, "found_only_action_requires_nested"):
            validate_ordered_task_plan(text, plan)

    def test_later_invalid_step_rejects_the_entire_plan(self):
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("inspect_camera", {"save_photo": "true"}, "看看"),
        ]}
        with self.assertRaisesRegex(ValueError, "step_2_invalid_arguments"):
            validate_ordered_task_plan("低头然后看看", plan)

    def test_invented_grounding_and_recursive_plans_are_rejected(self):
        plan = {"steps": [
            step("play_sequence", {"sequence_name": "head_down"}, "低头"),
            step("inspect_camera", {}, "拍照"),
        ]}
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            validate_ordered_task_plan("低头然后看看", plan)
        plan["steps"][1] = step("run_ordered_task", {}, "看看")
        with self.assertRaisesRegex(ValueError, "unknown_tool"):
            normalize_ordered_task_plan(plan)

    def test_step_limits_and_nested_search_arguments_are_bounded(self):
        item = step("inspect_camera", {}, "看看")
        for count in (0, 9):
            with self.assertRaisesRegex(ValueError, "steps_invalid"):
                normalize_ordered_task_plan({"steps": [copy.deepcopy(item) for _ in range(count)]})
        schema = ordered_task_tool_schema()["properties"]["steps"]
        self.assertEqual(schema["maxItems"], 8)
        self.assertEqual(schema["minItems"], 1)
        self.assertEqual(schema["items"]["properties"]["arguments"]["properties"]
                         ["on_found_actions"]["maxItems"], 7)
