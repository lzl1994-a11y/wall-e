import unittest
from unittest.mock import MagicMock

from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.dialog_tool_router import DialogToolRouter
from services.visual_search import VISUAL_SEARCH_TOOL_NAME


class DialogToolRouterTests(unittest.TestCase):
    def setUp(self):
        self.inspect = MagicMock(return_value="camera answer")
        self.conditional = MagicMock(return_value="conditional answer")
        self.visual_search = MagicMock(return_value={"status": "completed"})
        self.execute = MagicMock(return_value={"status": "completed"})
        self.router = DialogToolRouter(
            inspect_camera=self.inspect,
            run_conditional_task=self.conditional,
            run_visual_search=self.visual_search,
            execute_action=self.execute,
        )

    def test_routes_specialized_tools_without_running_regular_action(self):
        arguments = {"question": "前面有什么"}

        self.assertEqual(self.router.dispatch("inspect_camera", arguments), "camera answer")
        self.assertEqual(
            self.router.dispatch(CONDITIONAL_TASK_TOOL_NAME, arguments),
            "conditional answer",
        )
        self.assertEqual(
            self.router.dispatch(VISUAL_SEARCH_TOOL_NAME, arguments),
            {"status": "completed"},
        )
        self.execute.assert_not_called()

    def test_rejects_invalid_regular_action_before_execute_callback(self):
        result = self.router.dispatch("move_chassis", {"direction": "up"})

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["action"], "move_chassis")
        self.execute.assert_not_called()

    def test_passes_valid_regular_action_through_unchanged(self):
        arguments = {"direction": "forward", "duration": 1}

        result = self.router.dispatch("move_chassis", arguments)

        self.assertEqual(result, {"status": "completed"})
        self.execute.assert_called_once_with("move_chassis", arguments)
