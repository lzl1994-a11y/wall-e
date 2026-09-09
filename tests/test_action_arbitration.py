import unittest

from services.action_arbitration import ActionArbiter, source_priority


class ActionArbiterTests(unittest.TestCase):
    def test_independent_skill_owners_can_run_concurrently(self):
        arbiter = ActionArbiter()
        self.assertTrue(arbiter.submit("motion", "play_sequence", "llm_dialog").accepted)
        self.assertTrue(arbiter.submit("music", "control_music", "mcp").accepted)
        self.assertEqual({lease.request_id for lease in arbiter.active_leases}, {"motion", "music"})

    def test_lower_priority_conflict_is_rejected(self):
        arbiter = ActionArbiter()
        self.assertTrue(arbiter.submit("joy", "manual_servo", "joystick").accepted)
        decision = arbiter.submit("llm", "play_sequence", "native_behavior_tree")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "resource_busy:joystick:manual_servo")

    def test_equal_or_higher_priority_preempts_conflicting_lease(self):
        arbiter = ActionArbiter()
        arbiter.submit("old", "play_sequence", "llm_dialog")
        decision = arbiter.submit("new", "move_chassis", "native_behavior_tree")
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.preempted, ("old",))
        self.assertEqual([lease.request_id for lease in arbiter.active_leases], ["new"])

    def test_stop_all_preempts_every_owner(self):
        arbiter = ActionArbiter()
        arbiter.submit("motion", "play_sequence", "llm_dialog")
        arbiter.submit("music", "control_music", "mcp")
        decision = arbiter.submit("stop", "stop_all", "legacy")
        self.assertTrue(decision.accepted)
        self.assertEqual(set(decision.preempted), {"motion", "music"})
        self.assertEqual(source_priority("legacy", "stop_all"), 1000)

    def test_terminal_release_and_timeout_remove_leases(self):
        arbiter = ActionArbiter(lease_timeout=2)
        arbiter.submit("one", "play_sequence", "llm_dialog", now=10)
        self.assertTrue(arbiter.release("one"))
        arbiter.submit("two", "play_sequence", "llm_dialog", now=20)
        self.assertEqual(arbiter.expire(now=22), ("two",))

    def test_unknown_and_duplicate_requests_fail_closed(self):
        arbiter = ActionArbiter()
        self.assertFalse(arbiter.submit("unknown", "shell", "mcp").accepted)
        self.assertTrue(arbiter.submit("same", "play_sequence", "mcp").accepted)
        duplicate = arbiter.submit("same", "control_music", "mcp")
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate_request_id")


if __name__ == "__main__":
    unittest.main()
