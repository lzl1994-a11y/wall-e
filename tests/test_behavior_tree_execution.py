import threading
import unittest

from services.action_plan import compile_action_plan
from services.behavior_tree_execution import CorrelatedPlanExecutor
from services.behavior_tree_protocol import (
    BEHAVIOR_TREE_CANCEL_TOPIC,
    BEHAVIOR_TREE_EXECUTE_TOPIC,
    BEHAVIOR_TREE_STATUS_TOPIC,
    encode_plan_cancel,
    encode_plan_request,
    parse_plan_status,
)
from services.behavior_tree_workflow import NativeBehaviorTreeWorkflow


def make_plan():
    return compile_action_plan(
        turn_id="turn",
        user_prompt="挥手",
        actions=[{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}],
    )


class BehaviorTreeProtocolTests(unittest.TestCase):
    def test_topic_names_are_absolute_and_status_parser_is_strict(self):
        self.assertEqual(BEHAVIOR_TREE_EXECUTE_TOPIC, "/behavior_tree/execute")
        self.assertEqual(BEHAVIOR_TREE_CANCEL_TOPIC, "/behavior_tree/cancel")
        self.assertEqual(BEHAVIOR_TREE_STATUS_TOPIC, "/behavior_tree/status")
        self.assertIsNone(parse_plan_status("{}"))
        parsed = parse_plan_status({
            "plan_id": "plan-1",
            "status": "success",
            "results": [{"status": "completed"}],
        })
        self.assertEqual(parsed["status"], "success")

    def test_request_and_cancel_include_the_correlated_plan_id(self):
        plan = make_plan()
        self.assertIn(plan.plan_id, encode_plan_request(plan))
        self.assertIn(plan.plan_id, encode_plan_cancel(plan.plan_id))


class CorrelatedPlanExecutorTests(unittest.TestCase):
    def test_deadline_requests_correlated_cancel_without_claiming_halt(self):
        plan = make_plan()
        cancellations = []
        result = CorrelatedPlanExecutor().try_execute(
            plan, publish=lambda _: None, cancel_publish=cancellations.append,
            owner_available=lambda: True, timeout=0.01,
        )
        self.assertEqual(cancellations, [encode_plan_cancel(plan.plan_id)])
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["error"], "native_plan_timeout")

    def test_cancel_publish_failure_is_visible(self):
        def fail(_payload):
            raise RuntimeError("transport unavailable")

        result = CorrelatedPlanExecutor().try_execute(
            make_plan(), publish=lambda _: None, cancel_publish=fail,
            owner_available=lambda: True, timeout=0.01,
        )
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["error"], "native_plan_timeout:cancel_publish_failed")

    def test_user_cancel_is_not_resent_at_deadline(self):
        cancellations = []
        result = CorrelatedPlanExecutor().try_execute(
            make_plan(), publish=lambda _: None, cancel_publish=cancellations.append,
            owner_available=lambda: True, timeout=0.01, cancelled=lambda: True,
        )
        self.assertEqual(len(cancellations), 1)
        self.assertEqual(result["error"], "native_cancel_timeout")

    def test_absent_native_owner_returns_none_before_publishing(self):
        published = []
        result = CorrelatedPlanExecutor().try_execute(
            make_plan(),
            publish=published.append,
            cancel_publish=lambda _payload: None,
            owner_available=lambda: False,
            timeout=0.1,
        )
        self.assertIsNone(result)
        self.assertEqual(published, [])

    def test_waits_for_matching_terminal_plan_status(self):
        executor = CorrelatedPlanExecutor()
        plan = make_plan()

        def publish(_payload):
            executor.accept_status({"plan_id": "other", "status": "success", "results": []})
            executor.accept_status({"plan_id": plan.plan_id, "status": "accepted", "results": []})
            threading.Timer(0.02, lambda: executor.accept_status({
                "plan_id": plan.plan_id,
                "status": "success",
                "results": [{"status": "completed", "action": "play_sequence"}],
            })).start()

        result = executor.try_execute(
            plan,
            publish=publish,
            cancel_publish=lambda _payload: None,
            owner_available=lambda: True,
            timeout=1.0,
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["stopped"])
        self.assertEqual(result["results"][0]["status"], "completed")

    def test_cancellation_is_forwarded_and_does_not_fallback(self):
        executor = CorrelatedPlanExecutor()
        plan = make_plan()
        cancelled = threading.Event()
        cancel_payloads = []

        def publish(_payload):
            cancelled.set()

        def cancel_publish(payload):
            cancel_payloads.append(payload)
            executor.accept_status({
                "plan_id": plan.plan_id,
                "status": "halted",
                "results": [{"status": "interrupted"}],
                "error": "plan_cancelled",
            })

        result = executor.try_execute(
            plan,
            publish=publish,
            cancel_publish=cancel_publish,
            owner_available=lambda: True,
            timeout=1.0,
            cancelled=cancelled.is_set,
        )
        self.assertEqual(len(cancel_payloads), 1)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["error"], "plan_cancelled")


class NativeBehaviorTreeWorkflowTests(unittest.TestCase):
    def test_each_step_is_authorized_against_its_grounding(self):
        authorized = []
        submitted = []
        workflow = NativeBehaviorTreeWorkflow(
            authorize=lambda prompt, name, arguments: (
                authorized.append((prompt, name, arguments)) or (True, "")
            ),
            execute_plan=lambda plan: submitted.append(plan) or None,
        )

        result = workflow.invoke(
            turn_id="turn-grounded",
            user_prompt="把手举起来，然后抬一下头",
            actions=[
                {
                    "name": "play_sequence",
                    "arguments": {"sequence_name": "raise_hand"},
                    "grounding": "把手举起来",
                },
                {
                    "name": "play_sequence",
                    "arguments": {"sequence_name": "basic_nod"},
                    "grounding": "抬一下头",
                },
            ],
        )

        self.assertIsNone(result)
        self.assertEqual(
            [item[0] for item in authorized],
            ["把手举起来", "抬一下头"],
        )
        self.assertEqual(len(submitted), 1)

    def test_non_dictionary_native_result_is_normalized(self):
        for malformed in ([], "invalid", 42):
            with self.subTest(result=malformed):
                workflow = NativeBehaviorTreeWorkflow(
                    authorize=lambda *_: (True, ""),
                    execute_plan=lambda _: malformed,
                )
                result = workflow.invoke(
                    turn_id="turn", user_prompt="挥手",
                    actions=[{"name": "play_sequence", "arguments": {}}],
                )
                self.assertEqual(result["status"], "failure")
                self.assertEqual(result["error"], "invalid_native_plan_result")

    def test_authorizes_before_submitting_to_native_owner(self):
        submitted = []
        workflow = NativeBehaviorTreeWorkflow(
            authorize=lambda _prompt, _name, _arguments: (True, ""),
            execute_plan=lambda plan: submitted.append(plan) or {
                "plan_id": plan.plan_id,
                "status": "success",
                "results": [{
                    **plan.steps[0].to_dict(),
                    "status": "completed",
                    "action": plan.steps[0].name,
                }],
                "stopped": False,
            },
        )
        result = workflow.invoke(
            turn_id="turn", user_prompt="挥手", actions=[{
                "name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}
            }]
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(submitted), 1)

    def test_rejection_never_submits_any_part_of_the_plan(self):
        submitted = []
        workflow = NativeBehaviorTreeWorkflow(
            authorize=lambda _prompt, name, _arguments: (
                name != "control_music", "unsafe"
            ),
            execute_plan=lambda plan: submitted.append(plan),
        )
        result = workflow.invoke(
            turn_id="turn",
            user_prompt="do it",
            actions=[
                {"name": "play_sequence", "arguments": {
                    "sequence_name": "wave_hello"
                }},
                {"name": "control_music", "arguments": {"action": "play"}},
            ],
        )
        self.assertEqual(submitted, [])
        self.assertEqual(
            [item["status"] for item in result["results"]],
            ["skipped", "rejected"],
        )

    def test_missing_native_results_becomes_visible_failure(self):
        workflow = NativeBehaviorTreeWorkflow(
            authorize=lambda *_args: (True, ""),
            execute_plan=lambda plan: {
                "plan_id": plan.plan_id,
                "status": "rejected",
                "results": [],
                "error": "behavior_tree_busy",
            },
        )
        result = workflow.invoke(
            turn_id="turn", user_prompt="挥手", actions=[{
                "name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}
            }]
        )
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["results"][0]["status"], "rejected")
        self.assertEqual(result["error"], "behavior_tree_busy")


class NativePackageContractTests(unittest.TestCase):
    def test_native_node_uses_stateful_bt_and_correlated_topics(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = (root / "cpp_nodes" / "wali_behavior_tree" / "src" /
                  "behavior_tree_node.cpp").read_text(encoding="utf-8")
        self.assertIn("public BT::StatefulActionNode", source)
        self.assertIn('constexpr char kExecuteTopic[] = "/behavior_tree/execute"', source)
        self.assertIn('constexpr char kActionRequestTopic[] = "/action_request"', source)
        self.assertIn('constexpr char kActionStatusTopic[] = "/action_status"', source)
        self.assertIn("kMaxPlanSteps = 8", source)
        self.assertIn("publish_emergency_stop_once", source)
        self.assertIn("load_skill_registry", source)
        self.assertIn('definition["plan_resources"]', source)
        self.assertNotIn("kActionResources", source)
        self.assertIn("<Fallback", source)
        self.assertIn("<RetryUntilSuccessful", source)
        self.assertIn("<Timeout msec=", source)
        self.assertIn("RecoveryStopNode", source)

    def test_cmake_handles_humble_multiarch_behavior_tree_package(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cmake = (root / "cpp_nodes" / "wali_behavior_tree" /
                 "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("${prefix}/lib/${CMAKE_LIBRARY_ARCHITECTURE}", cmake)
        self.assertIn("find_library(BTCPP_LIBRARY behaviortree_cpp", cmake)
        self.assertNotIn("find_package(behaviortree_cpp REQUIRED)", cmake)

    def test_launcher_prefers_the_repository_local_binary(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        launcher = (root / "nodes" / "native_behavior_tree_launcher.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('/ "install"', launcher)
        self.assertIn("local_binary.is_file()", launcher)
        self.assertIn('Path("/opt/ros") / ros_distro / "setup.bash"', launcher)
        self.assertIn('source "$1" && exec "$2"', launcher)
        self.assertIn("skill_registry_path:=", launcher)
        self.assertLess(launcher.index("local_binary.is_file()"), launcher.index("shutil.which"))


if __name__ == "__main__":
    unittest.main()
