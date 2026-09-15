from services.dialog_action_execution import DialogActionExecutor


class _Executor:
    def __init__(self):
        self.calls = []
        self.statuses = []

    def execute(self, name, arguments, **kwargs):
        self.calls.append((name, arguments, kwargs))
        return {"status": "success"}

    def accept_status(self, payload):
        self.statuses.append(payload)
        return True


def test_dialog_action_executor_binds_transport_and_preserves_source():
    executor = _Executor()
    published = []
    owner_available = lambda: True
    adapter = DialogActionExecutor(
        executor=executor,
        publish=published.append,
        owner_available=owner_available,
        timeout=20.0,
    )

    result = adapter.execute(
        "play_sequence", {"sequence_name": "wave_hello"}, source="voice_dialog"
    )

    assert result == {"status": "success"}
    name, arguments, kwargs = executor.calls[0]
    assert (name, arguments) == ("play_sequence", {"sequence_name": "wave_hello"})
    assert kwargs["owner_available"] is owner_available
    assert kwargs["timeout"] == 20.0
    assert kwargs["source"] == "voice_dialog"
    kwargs["publish"]("request")
    assert published == ["request"]


def test_dialog_action_executor_forwards_statuses():
    executor = _Executor()
    adapter = DialogActionExecutor(
        executor=executor,
        publish=lambda _payload: None,
        owner_available=lambda: True,
    )

    assert adapter.accept_status('{"status":"success"}') is True
    assert executor.statuses == ['{"status":"success"}']
