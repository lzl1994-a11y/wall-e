import json
import unittest

from services.action_command import build_action_cmd, parse_action_cmd, parse_action_request
from services.action_status import build_action_status, parse_action_status


class ActionEnvelopeTests(unittest.TestCase):
    def test_correlated_envelope_remains_legacy_parser_compatible(self):
        payload = build_action_cmd(
            "move_chassis",
            {"direction": "forward", "duration": 1},
            request_id="req-1",
            source="mcp",
        )

        self.assertEqual(
            parse_action_cmd(payload),
            ("move_chassis", {"direction": "forward", "duration": 1}),
        )
        self.assertEqual(
            parse_action_request(payload),
            {
                "name": "move_chassis",
                "arguments": {"direction": "forward", "duration": 1},
                "request_id": "req-1",
                "source": "mcp",
            },
        )

    def test_invalid_correlation_metadata_is_rejected(self):
        payload = json.dumps({"name": "stop_all", "arguments": {}, "request_id": ""})
        self.assertIsNone(parse_action_request(payload))


class ActionStatusTests(unittest.TestCase):
    def test_status_round_trip(self):
        payload = build_action_status(
            "req-1",
            "move_chassis",
            "completed",
            source="sequence_ros_node",
        )
        self.assertEqual(parse_action_status(payload)["status"], "completed")

    def test_unknown_status_is_rejected(self):
        payload = json.dumps({
            "request_id": "req-1",
            "name": "move_chassis",
            "status": "probably_done",
        })
        self.assertIsNone(parse_action_status(payload))


if __name__ == "__main__":
    unittest.main()
