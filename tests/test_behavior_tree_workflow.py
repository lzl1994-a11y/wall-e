import unittest

from services.action_plan import PlanValidationError, compile_action_plan
from services.behavior_tree_workflow import BehaviorTreeActionWorkflow


class ActionPlanTests(unittest.TestCase):
    def test_compiler_owns_ids_dependencies_and_resources(self):
        plan = compile_action_plan(
            turn_id="turn-1",
            user_prompt="先挥手再播放音乐",
            actions=[
                {"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}},
                {"name": "control_music", "arguments": {"command": "play"}},
            ],
        )

        self.assertTrue(plan.plan_id.startswith("plan-"))
        self.assertEqual([step.step_id for step in plan.steps], ["step-01", "step-02"])
        self.assertEqual(plan.steps[0].depends_on, ())
        self.assertEqual(plan.steps[1].depends_on, ("step-01",))
        self.assertEqual(plan.steps[0].resources, ("servo_motion",))
        self.assertEqual(plan.steps[1].resources, ("audio_music", "display"))
        self.assertEqual(plan.schema_version, 1)
        self.assertEqual(plan.root_type, "Sequence")
        self.assertEqual(plan.on_failure, "stop_remaining")

    def test_model_cannot_override_server_owned_orchestration_fields(self):
        plan = compile_action_plan(
            turn_id="trusted-turn",
            user_prompt="挥手",
            actions=[{
                "name": "play_sequence",
                "arguments": {"sequence_name": "wave_hello"},
                "step_id": "model-step",
                "depends_on": ["anything"],
                "resources": ["untrusted"],
                "on_failure": "continue",
            }],
        )

        self.assertEqual(plan.steps[0].step_id, "step-01")
        self.assertEqual(plan.steps[0].depends_on, ())
        self.assertEqual(plan.steps[0].resources, ("servo_motion",))
        self.assertEqual(plan.on_failure, "stop_remaining")

    def test_rejects_empty_too_large_and_malformed_plans(self):
        invalid = (
            [],
            [{"name": "x", "arguments": {}}] * 9,
            [{"name": "", "arguments": {}}],
            [{"name": "x", "arguments": []}],
        )
        for actions in invalid:
            with self.subTest(actions=actions):
                with self.assertRaises(PlanValidationError):
                    compile_action_plan(turn_id="turn", user_prompt="do it", actions=actions)


class BehaviorTreeActionWorkflowTests(unittest.TestCase):
    ACTIONS = [
        {"name": "first", "arguments": {"value": 1}},
        {"name": "second", "arguments": {"value": 2}},
    ]

    def workflow(self, *, authorize=None, execute=None, cancelled=None):
        return BehaviorTreeActionWorkflow(
            authorize=authorize or (lambda _prompt, _name, _arguments: (True, "")),
            execute=execute or (lambda name, _arguments: {"status": "completed", "action": name}),
            cancelled=cancelled,
        )

    def test_sequence_executes_in_order_and_reports_plan(self):
        calls = []
        workflow = self.workflow(execute=lambda name, arguments: (
            calls.append((name, arguments))
            or {"status": "completed", "request_id": f"req-{len(calls)}"}
        ))

        state = workflow.invoke(
            turn_id="turn", user_prompt="first then second", actions=self.ACTIONS
        )

        self.assertEqual(calls, [("first", {"value": 1}), ("second", {"value": 2})])
        self.assertEqual(state["status"], "success")
        self.assertFalse(state["stopped"])
        self.assertEqual(state["plan_id"], state["plan"]["plan_id"])
        self.assertEqual([item["node_status"] for item in state["results"]], ["success", "success"])

    def test_failure_rejection_and_interruption_stop_remaining(self):
        cases = (
            (lambda *_args: (True, ""), lambda *_args: {"status": "failed", "reason": "motor"}, None, "failed"),
            (lambda *_args: (False, "unsafe"), None, None, "rejected"),
            (lambda *_args: (True, ""), None, lambda: True, "interrupted"),
        )
        for authorize, execute, cancelled, expected in cases:
            with self.subTest(expected=expected):
                calls = []
                runner = execute or (lambda name, _args: calls.append(name) or {"status": "completed"})
                workflow = self.workflow(authorize=authorize, execute=runner, cancelled=cancelled)
                state = workflow.invoke(
                    turn_id="turn", user_prompt="do it", actions=self.ACTIONS
                )
                self.assertEqual([item["status"] for item in state["results"]], [expected, "skipped"])
                self.assertTrue(state["stopped"])
                self.assertIn(state["status"], ("failure", "halted"))

    def test_invalid_plan_fails_closed_without_execution(self):
        calls = []
        state = self.workflow(
            execute=lambda name, _args: calls.append(name)
        ).invoke(turn_id="turn", user_prompt="", actions=[])

        self.assertEqual(calls, [])
        self.assertEqual(state["status"], "failure")
        self.assertEqual(state["results"][0]["status"], "rejected")
        self.assertEqual(state["error"], "action_plan_empty")

    def test_failure_after_success_reports_the_actual_failure_reason(self):
        calls = []

        def execute(name, _arguments):
            calls.append(name)
            if name == "first":
                return {"status": "completed"}
            return {"status": "timeout", "reason": "no_terminal_executor_status"}

        state = self.workflow(execute=execute).invoke(
            turn_id="turn", user_prompt="do it", actions=self.ACTIONS
        )

        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(state["status"], "failure")
        self.assertEqual(state["error"], "no_terminal_executor_status")

    def test_authorizer_exception_fails_closed(self):
        state = self.workflow(
            authorize=lambda *_args: (_ for _ in ()).throw(RuntimeError("guard error"))
        ).invoke(turn_id="turn", user_prompt="do it", actions=self.ACTIONS)

        self.assertEqual(state["results"][0]["status"], "rejected")
        self.assertEqual(state["error"], "authorization_failed:guard error")


if __name__ == "__main__":
    unittest.main()
