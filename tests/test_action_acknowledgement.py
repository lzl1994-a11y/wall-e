import unittest

from services.action_acknowledgement import action_acknowledgement


class ActionAcknowledgementTests(unittest.TestCase):
    def test_known_actions_have_non_speculative_acknowledgements(self):
        self.assertEqual(
            action_acknowledgement([{
                "name": "play_sequence",
                "arguments": '{"sequence_name":"turn_head_right"}',
            }]),
            "好的，我向右看。",
        )
        self.assertEqual(
            action_acknowledgement([{
                "name": "set_tracking_mode",
                "arguments": {"mode": "follow_me"},
            }]),
            "好的，我跟着你。",
        )
        self.assertEqual(
            action_acknowledgement([{
                "name": "move_chassis",
                "arguments": {"direction": "forward", "duration": 2},
            }]),
            "好的，我向前走。",
        )

    def test_unknown_or_multiple_actions_use_generic_acceptance(self):
        self.assertEqual(
            action_acknowledgement([{"name": "unknown", "arguments": {}}]),
            "好的。",
        )
        self.assertEqual(
            action_acknowledgement([
                {"name": "play_sequence", "arguments": {}},
                {"name": "express_emotion", "arguments": {}},
            ]),
            "好的。",
        )
        self.assertEqual(action_acknowledgement([]), "")


if __name__ == "__main__":
    unittest.main()
