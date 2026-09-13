import json

from services.action_plan import compile_action_plan
from services.ros2_action_execution import Ros2ActionPlanExecutor


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _Goal:
    def __init__(self):
        self.target_tree = ""
        self.payload = ""


class _GoalType:
    Goal = _Goal


class _GoalHandle:
    accepted = True

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.cancelled = False

    def get_result_async(self):
        return _Future(self._wrapped)

    def cancel_goal_async(self):
        self.cancelled = True
        return _Future(None)


class _Client:
    def __init__(self, goal_handle=None, available=True):
        self.goal_handle = goal_handle
        self.available = available
        self.sent = None

    def wait_for_server(self, timeout_sec):
        return self.available

    def send_goal_async(self, goal):
        self.sent = goal
        return _Future(self.goal_handle)


def _plan():
    return compile_action_plan(
        turn_id="turn",
        user_prompt="挥手",
        actions=[{"name": "play_sequence", "arguments": {"sequence_name": "wave_hello"}}],
    )


def test_ros2_action_executor_returns_none_only_before_submission():
    executor = Ros2ActionPlanExecutor(_Client(available=False), _GoalType)
    assert executor.try_execute(_plan()) is None


def test_ros2_action_executor_maps_standard_result_to_plan_status():
    plan = _plan()
    payload = json.dumps({
        "plan_id": plan.plan_id,
        "status": "success",
        "results": [],
        "source": "native_behavior_tree",
    })
    wrapped = type("Wrapped", (), {
        "result": type("Result", (), {"return_message": payload})()
    })()
    client = _Client(_GoalHandle(wrapped))
    result = Ros2ActionPlanExecutor(client, _GoalType).try_execute(plan)

    assert client.sent.target_tree == "WaliTask"
    assert json.loads(client.sent.payload)["plan_id"] == plan.plan_id
    assert result["status"] == "success"
    assert result["source"] == "native_behavior_tree"
