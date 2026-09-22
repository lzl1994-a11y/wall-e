import json
import unittest

from services.vision.visual_search import (
    VISUAL_SEARCH_TOOL_NAME,
    VisualSearchWorkflow,
    VisualSearchViewWorkflow,
    compile_visual_search_plan,
    encode_visual_search_status,
    normalize_visual_search_arguments,
    parse_visual_search_request,
    parse_visual_search_result,
)


class VisualSearchTests(unittest.TestCase):
    def test_compiles_native_visual_search_plan(self):
        plan = compile_visual_search_plan(
            turn_id="turn-1",
            arguments={
                "target": "红色箱子",
                "question": "红色箱子在哪里",
                "search_direction": "spin",
                "motion_duration": 1,
                "max_views": 4,
            },
        )

        payload = plan.to_dict()
        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["root_type"], "VisualSearch")
        self.assertEqual(payload["on_failure"], "report_not_found")
        self.assertEqual(payload["max_views"], 4)
        self.assertEqual(payload["steps"][0]["name"], "move_chassis")
        self.assertEqual(payload["steps"][0]["max_attempts"], 3)

    def test_arguments_are_bounded_without_object_allowlist(self):
        normalized = normalize_visual_search_arguments({"target": "电饭煲"})
        self.assertEqual(normalized["target"], "电饭煲")
        self.assertEqual(normalized["max_views"], 3)
        with self.assertRaisesRegex(ValueError, "views"):
            normalize_visual_search_arguments({"target": "人", "max_views": 20})

    def test_compiles_model_proposed_actions_after_a_successful_search(self):
        plan = compile_visual_search_plan(
            turn_id="turn-then",
            arguments={
                "target": "红色箱子",
                "on_found_actions": [{
                    "name": "play_sequence",
                    "arguments": {"sequence_name": "wave_hello"},
                }],
            },
        )

        payload = plan.to_dict()
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["root_type"], "VisualSearchThen")
        self.assertEqual(payload["steps"][1]["name"], "play_sequence")
        self.assertEqual(payload["steps"][1]["depends_on"], ["step-01"])
        self.assertEqual(payload["on_found_actions"][0]["name"], "play_sequence")

    def test_completion_actions_reject_unknown_or_open_shapes(self):
        with self.assertRaisesRegex(ValueError, "completion_action"):
            normalize_visual_search_arguments({
                "target": "箱子", "on_found_actions": [{"name": "shell"}],
            })
        with self.assertRaisesRegex(ValueError, "unknown"):
            normalize_visual_search_arguments({
                "target": "箱子",
                "on_found_actions": [{"name": "search_environment", "arguments": {}}],
            })

    def test_leaf_protocol_is_correlated(self):
        request = {
            "request_id": "req-1",
            "plan_id": "plan-1",
            "target": "用户",
            "question": "用户在哪里",
            "attempt": 2,
            "max_views": 3,
        }
        self.assertEqual(parse_visual_search_request(json.dumps(request)), request)
        payload = encode_visual_search_status(request, {
            "status": "found",
            "evidence": "画面左侧有人",
            "response": "你在我的左前方。",
        })
        self.assertEqual(json.loads(payload)["request_id"], "req-1")
        self.assertEqual(
            parse_visual_search_result(json.loads(payload))["status"], "found"
        )

    def test_result_parser_rejects_open_status(self):
        self.assertIsNone(parse_visual_search_result({
            "status": "maybe", "evidence": "", "response": ""
        }))

    def test_execution_workflow_validates_completion_actions_and_normalizes_result(self):
        workflow = VisualSearchWorkflow(authorize=lambda _name, _arguments: (True, ""))
        prepared = workflow.prepare(
            turn_id="turn-1",
            arguments={
                "target": "箱子",
                "on_found_actions": [{
                    "name": "play_sequence",
                    "arguments": {"sequence_name": "wave_hello"},
                }],
            },
        )

        result = workflow.complete(prepared.plan, {
            "status": "success",
            "results": [{
                "action": VISUAL_SEARCH_TOOL_NAME,
                "status": "found",
                "found": True,
                "attempts": 2,
            }],
        })

        self.assertIsNone(prepared.rejection)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["found"])

    def test_execution_workflow_rejects_invalid_completion_and_missing_executor(self):
        workflow = VisualSearchWorkflow(
            authorize=lambda _name, _arguments: (False, "invalid_arguments")
        )
        prepared = workflow.prepare(
            turn_id="turn-1",
            arguments={
                "target": "箱子",
                "on_found_actions": [{
                    "name": "play_sequence",
                    "arguments": {"sequence_name": "wave_hello"},
                }],
            },
        )
        plan = compile_visual_search_plan(
            turn_id="turn-2", arguments={"target": "箱子"}
        )

        self.assertEqual(prepared.rejection["status"], "rejected")
        self.assertEqual(
            workflow.complete(plan, None)["reason"],
            "native_behavior_tree_unavailable",
        )

    def test_view_workflow_evaluates_captured_frame_and_handles_unavailable_view(self):
        preview = type("Preview", (), {"busy": False, "last_frame": b"jpeg", "error": ""})()
        workflow = VisualSearchViewWorkflow(
            capture=lambda: preview,
            evaluate=lambda target, question, image: {
                "status": "found",
                "evidence": f"{target}:{image}",
                "response": question,
            },
        )
        request = {"target": "杯子", "question": "杯子在哪"}

        decision = workflow.invoke(request)

        self.assertEqual(decision.result["status"], "found")
        self.assertIn("anBlZw==", decision.result["evidence"])
        unavailable = VisualSearchViewWorkflow(
            capture=lambda: type("Preview", (), {
                "busy": True, "last_frame": None, "error": "camera_busy"
            })(),
            evaluate=lambda *_args: self.fail("must not evaluate an unavailable frame"),
        ).invoke(request)
        self.assertEqual(unavailable.result["evidence"], "camera_busy")

    def test_view_workflow_returns_safe_result_when_evaluation_fails(self):
        workflow = VisualSearchViewWorkflow(
            capture=lambda: type("Preview", (), {
                "busy": False, "last_frame": b"jpeg", "error": ""
            })(),
            evaluate=lambda *_args: (_ for _ in ()).throw(RuntimeError("model unavailable")),
        )

        decision = workflow.invoke({"target": "杯子", "question": "在哪"})

        self.assertEqual(decision.result["evidence"], "visual_search_view_failed")
        self.assertEqual(decision.error, "model unavailable")


if __name__ == "__main__":
    unittest.main()
