"""ROS-independent adapter for dialogue action-request execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class DialogActionExecutor:
    """Bind one correlated action executor to dialogue transport callbacks."""

    def __init__(
        self,
        *,
        executor: Any | None,
        publish: Callable[[str], None],
        owner_available: Callable[[], bool],
        timeout: float = 20.0,
    ) -> None:
        self._executor = executor
        self._publish = publish
        self._owner_available = owner_available
        self._timeout = timeout

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        source: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        return self._executor.execute(
            name,
            arguments,
            publish=self._publish,
            owner_available=self._owner_available,
            timeout=self._timeout,
            source=source,
            cancelled=cancelled,
        )

    def accept_status(self, payload: Any) -> bool:
        if self._executor is None:
            return False
        return self._executor.accept_status(payload)


__all__ = ["DialogActionExecutor"]
