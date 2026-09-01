import unittest
from pathlib import Path

import yaml

from services.action_intent_guard import validate_action_call


ROOT = Path(__file__).resolve().parents[1]
EXPRESSION_NAMES = {
    "expression_neutral",
    "expression_listening",
    "expression_thinking",
    "expression_happy",
    "expression_sad",
    "expression_surprised",
    "expression_confused",
    "expression_concerned",
}


class ExpressionLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            (ROOT / "core" / "config.yaml").read_text(encoding="utf-8")
        )
        cls.library = yaml.safe_load(
            (ROOT / "core" / "sequences.yaml").read_text(encoding="utf-8")
        )
        cls.servos = {
            servo["name"]: servo for servo in cls.config["servos"]
        }

    def test_expression_candidates_are_registered_as_poses(self):
        self.assertTrue(EXPRESSION_NAMES <= set(self.library["poses"]))

    def test_expression_targets_stay_within_configured_servo_limits(self):
        for name in EXPRESSION_NAMES:
            with self.subTest(expression=name):
                for servo_name, target in self.library["poses"][name]["targets"].items():
                    servo = self.servos[servo_name]
                    self.assertGreaterEqual(target, min(servo["limit_1"], servo["limit_2"]))
                    self.assertLessEqual(target, max(servo["limit_1"], servo["limit_2"]))

    def test_surprised_uses_the_specified_maximum_neck_extension(self):
        targets = self.library["poses"]["expression_surprised"]["targets"]
        self.assertEqual(targets["neck_top"], self.servos["neck_top"]["limit_2"])
        self.assertEqual(targets["neck_bottom"], self.servos["neck_bottom"]["limit_2"])

    def test_sad_uses_the_specified_downward_neck_coupling(self):
        targets = self.library["poses"]["expression_sad"]["targets"]
        self.assertGreater(targets["neck_top"], self.servos["neck_top"]["init"])
        self.assertLess(targets["neck_bottom"], self.servos["neck_bottom"]["init"])

    def test_explicit_expression_requests_pass_the_action_guard(self):
        cases = (
            ("请做一个认真倾听的表情", "expression_listening"),
            ("请做出思考的表情", "expression_thinking"),
            ("请做个开心表情", "expression_happy"),
            ("请做个难过的表情", "expression_sad"),
            ("请做个惊讶的表情", "expression_surprised"),
            ("请做一个疑惑表情", "expression_confused"),
            ("请做出关切的表情", "expression_concerned"),
        )
        for user_text, expression in cases:
            with self.subTest(expression=expression):
                self.assertEqual(
                    validate_action_call(
                        user_text,
                        "play_sequence",
                        {"sequence_name": expression},
                    ),
                    (True, ""),
                )

    def test_negated_and_third_party_expression_requests_are_rejected(self):
        cases = (
            ("不要做疑惑表情", "expression_confused"),
            ("不要做惊讶表情", "expression_surprised"),
            ("他正在做思考表情", "expression_thinking"),
        )
        for user_text, expression in cases:
            with self.subTest(user_text=user_text):
                allowed, _reason = validate_action_call(
                    user_text,
                    "play_sequence",
                    {"sequence_name": expression},
                )
                self.assertFalse(allowed)
