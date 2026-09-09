import json
import unittest

from services.action_cancel import build_action_cancel, parse_action_cancel
from services.action_plan import compile_action_plan
from services.action_registry import (
    ACTION_SKILL_REGISTRY_PATH,
    get_action_skill,
    load_action_skills,
)


class ActionRegistryTests(unittest.TestCase):
    def test_registry_is_valid_and_plan_compiler_uses_it(self):
        raw = json.loads(ACTION_SKILL_REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(raw["skills"]), set(load_action_skills()))
        plan = compile_action_plan(
            turn_id="turn",
            user_prompt="move",
            actions=[{"name": "move_chassis", "arguments": {
                "direction": "forward", "duration": 1,
            }}],
        )
        self.assertEqual(
            plan.steps[0].resources,
            get_action_skill("move_chassis").plan_resources,
        )
        self.assertTrue(get_action_skill("move_chassis").supports_cancel)
        self.assertTrue(get_action_skill("move_chassis").action_bus)
        self.assertFalse(get_action_skill("inspect_camera").action_bus)

    def test_targeted_cancel_protocol_round_trip(self):
        payload = build_action_cancel(
            "old", "play_sequence",
            reason="preempted_by:joystick:manual_servo",
            replacement_request_id="new",
        )
        parsed = parse_action_cancel(payload)
        self.assertEqual(parsed["request_id"], "old")
        self.assertEqual(parsed["replacement_request_id"], "new")
        self.assertIsNone(parse_action_cancel("{}"))


if __name__ == "__main__":
    unittest.main()
