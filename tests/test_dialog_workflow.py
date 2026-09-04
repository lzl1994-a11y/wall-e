import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.dialog_workflow import CameraInspectionWorkflow, ConditionalTaskWorkflow


class CameraInspectionWorkflowTests(unittest.TestCase):
    def test_success_analyzes_frame_and_returns_one_answer(self):
        analyze = MagicMock(return_value="你手里拿着一个杯子。")
        workflow = CameraInspectionWorkflow(
            capture=lambda: SimpleNamespace(busy=False, last_frame=b"jpeg"),
            analyze=analyze,
        )

        result = workflow.invoke(turn_id="turn-1", user_prompt="看看我手里有什么")

        self.assertEqual(result["answer"], "你手里拿着一个杯子。")
        self.assertNotIn("error", result)
        analyze.assert_called_once_with(b"jpeg", "看看我手里有什么")

    def test_missing_frame_finishes_without_calling_model(self):
        analyze = MagicMock()
        workflow = CameraInspectionWorkflow(
            capture=lambda: SimpleNamespace(
                busy=False,
                last_frame=None,
                error="camera_timeout",
            ),
            analyze=analyze,
        )

        result = workflow.invoke(turn_id="turn-2", user_prompt="看看前面")

        self.assertEqual(result["error"], "camera_timeout")
        self.assertEqual(result["answer"], "我现在看不到画面，检查一下摄像头连接。")
        analyze.assert_not_called()

    def test_model_failure_becomes_fail_closed_user_answer(self):
        analyze = MagicMock(side_effect=RuntimeError("provider unavailable"))
        workflow = CameraInspectionWorkflow(
            capture=lambda: SimpleNamespace(busy=False, last_frame=b"jpeg"),
            analyze=analyze,
        )

        result = workflow.invoke(turn_id="turn-3", user_prompt="这是什么")

        self.assertEqual(result["error"], "provider unavailable")
        self.assertEqual(
            result["answer"],
            "这张图我没分析出来，你换个角度再让我看看。",
        )


class ConditionalTaskWorkflowTests(unittest.TestCase):
    PLAN = {
        "observation": "看看前面有什么",
        "condition": "画面中存在用户指定的目标",
        "action_name": "play_sequence",
        "action_arguments": {"sequence_name": "raise_hand"},
    }

    def workflow(self, *, decision, authorize=(True, ""), execute=None):
        execute = execute or MagicMock(return_value={
            "status": "completed",
            "action": "play_sequence",
            "request_id": "req-1",
        })
        return ConditionalTaskWorkflow(
            capture=lambda: SimpleNamespace(busy=False, last_frame=b"jpeg"),
            evaluate=MagicMock(return_value=decision),
            authorize=MagicMock(return_value=authorize),
            execute=execute,
        ), execute

    def test_yes_executes_once_and_requires_completed_status(self):
        workflow, execute = self.workflow(
            decision={"decision": "yes", "evidence": "目标清晰可见"}
        )

        result = workflow.invoke(
            turn_id="turn-yes", user_prompt="复合任务", plan=self.PLAN
        )

        self.assertEqual(result["decision"], "yes")
        self.assertEqual(result["action_result"]["status"], "completed")
        self.assertEqual(result["answer"], "条件满足，动作已经执行完成。")
        execute.assert_called_once_with(
            "play_sequence", {"sequence_name": "raise_hand"}
        )

    def test_no_and_uncertain_never_execute(self):
        for decision, expected in (
            ("no", "条件不满足，我没有执行动作。"),
            ("uncertain", "我没法确定条件是否满足，所以没有执行动作。"),
        ):
            with self.subTest(decision=decision):
                workflow, execute = self.workflow(
                    decision={"decision": decision, "evidence": ""}
                )
                result = workflow.invoke(
                    turn_id="turn-no", user_prompt="复合任务", plan=self.PLAN
                )
                self.assertEqual(result["answer"], expected)
                execute.assert_not_called()

    def test_invalid_model_output_fails_closed_without_action(self):
        workflow, execute = self.workflow(decision="当然可以")

        result = workflow.invoke(
            turn_id="turn-invalid", user_prompt="复合任务", plan=self.PLAN
        )

        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(result["evidence"], "invalid_model_output")
        execute.assert_not_called()

    def test_authorization_and_executor_failure_are_not_reported_as_success(self):
        workflow, execute = self.workflow(
            decision={"decision": "yes", "evidence": ""},
            authorize=(False, "blocked"),
        )
        denied = workflow.invoke(
            turn_id="turn-denied", user_prompt="复合任务", plan=self.PLAN
        )
        self.assertEqual(denied["error"], "blocked")
        execute.assert_not_called()

        failed_execute = MagicMock(return_value={
            "status": "timeout",
            "action": "play_sequence",
            "reason": "no_terminal_executor_status",
        })
        workflow, _ = self.workflow(
            decision={"decision": "yes", "evidence": ""},
            execute=failed_execute,
        )
        failed = workflow.invoke(
            turn_id="turn-failed", user_prompt="复合任务", plan=self.PLAN
        )
        self.assertEqual(failed["answer"], "条件满足，但动作没有执行成功。")
        self.assertEqual(failed["error"], "no_terminal_executor_status")


if __name__ == "__main__":
    unittest.main()
