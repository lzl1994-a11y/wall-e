import json
import unittest

from services.visual_search import (
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


if __name__ == "__main__":
    unittest.main()
