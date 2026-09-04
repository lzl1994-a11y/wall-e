import json
import unittest
from unittest.mock import patch

from services.action_execution import CorrelatedActionExecutor
from services.action_status import build_action_status


class CorrelatedActionExecutorTests(unittest.TestCase):
    def test_waits_for_matching_terminal_completion(self):
        executor = CorrelatedActionExecutor()
        published = []

        def publish(payload):
            published.append(json.loads(payload))
            request_id = published[-1]["request_id"]
            executor.accept_status(build_action_status(
                request_id,
                "play_sequence",
                "accepted",
                source="sequence_ros_node",
            ))
            executor.accept_status(build_action_status(
                request_id,
                "play_sequence",
                "completed",
                source="sequence_ros_node",
            ))

        with patch(
            "services.action_execution.new_action_request_id",
            return_value="workflow-request-1",
        ):
            result = executor.execute(
                "play_sequence",
                {"sequence_name": "raise_hand"},
                publish=publish,
                owner_available=lambda: True,
                timeout=0.2,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["request_id"], "workflow-request-1")
        self.assertEqual(published[0]["source"], "llm_workflow")

    def test_accepted_without_terminal_status_times_out(self):
        executor = CorrelatedActionExecutor()

        def publish(payload):
            request_id = json.loads(payload)["request_id"]
            executor.accept_status(build_action_status(
                request_id, "play_sequence", "accepted"
            ))

        result = executor.execute(
            "play_sequence",
            {"sequence_name": "raise_hand"},
            publish=publish,
            owner_available=lambda: True,
            timeout=0.1,
        )

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["last_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
