from services.orchestration.native_plan_execution import NativePlanExecutionAdapter


class _Executor:
    def __init__(self, result=None):
        self.result = result
        self.calls = []
        self.statuses = []

    def try_execute(self, plan, **kwargs):
        self.calls.append((plan, kwargs))
        return self.result

    def accept_status(self, payload):
        self.statuses.append(payload)
        return payload == "valid"


def test_prefers_ros2_action_result_over_legacy_transport():
    ros2 = _Executor({"status": "success"})
    legacy = _Executor({"status": "failure"})
    adapter = NativePlanExecutionAdapter(ros2_executor=ros2, legacy_executor=legacy)

    assert adapter.try_execute("plan", timeout=3.0, cancelled=lambda: False) == {
        "status": "success"
    }
    assert len(ros2.calls) == 1
    assert legacy.calls == []


def test_falls_back_to_legacy_with_transport_callbacks():
    ros2 = _Executor(None)
    legacy = _Executor({"status": "success"})
    published = []
    cancelled = []
    owner_available = lambda: True
    adapter = NativePlanExecutionAdapter(
        ros2_executor=ros2,
        legacy_executor=legacy,
        publish=published.append,
        cancel_publish=cancelled.append,
        owner_available=owner_available,
    )

    result = adapter.try_execute("plan", timeout=3.0, cancelled=lambda: False)

    assert result == {"status": "success"}
    plan, kwargs = legacy.calls[0]
    assert plan == "plan"
    assert kwargs["timeout"] == 3.0
    assert kwargs["owner_available"] is owner_available
    kwargs["publish"]("request")
    kwargs["cancel_publish"]("cancel")
    assert published == ["request"]
    assert cancelled == ["cancel"]


def test_returns_none_without_a_complete_legacy_transport():
    assert NativePlanExecutionAdapter(legacy_executor=_Executor()).try_execute("plan") is None


def test_forwards_legacy_status_when_configured():
    legacy = _Executor()
    adapter = NativePlanExecutionAdapter(legacy_executor=legacy)

    assert adapter.accept_status("valid") is True
    assert legacy.statuses == ["valid"]
    assert NativePlanExecutionAdapter().accept_status("valid") is False
