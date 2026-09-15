"""Select and coordinate native behavior-tree plan transports.

The decision is deliberately ROS-independent: nodes supply their Action client
adapter and legacy topic callbacks at the boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class NativePlanExecutionAdapter:
    """Prefer ROS 2 Actions and retain the correlated-topic fallback."""

    def __init__(
        self,
        *,
        ros2_executor: Any | None = None,
        legacy_executor: Any | None = None,
        publish: Callable[[str], None] | None = None,
        cancel_publish: Callable[[str], None] | None = None,
        owner_available: Callable[[], bool] | None = None,
    ):
        self._ros2_executor = ros2_executor
        self._legacy_executor = legacy_executor
        self._publish = publish
        self._cancel_publish = cancel_publish
        self._owner_available = owner_available

    def accept_status(self, payload: Any) -> bool:
        """Forward legacy status messages when that transport is configured."""
        if self._legacy_executor is None:
            return False
        return self._legacy_executor.accept_status(payload)

    def try_execute(
        self,
        plan: Any,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Execute via Action first, falling back only when it is unavailable."""
        if self._ros2_executor is not None:
            result = self._ros2_executor.try_execute(
                plan, timeout=timeout, cancelled=cancelled
            )
            if result is not None:
                return result
        if (
            self._legacy_executor is None
            or self._publish is None
            or self._cancel_publish is None
            or self._owner_available is None
        ):
            return None
        return self._legacy_executor.try_execute(
            plan,
            publish=self._publish,
            cancel_publish=self._cancel_publish,
            owner_available=self._owner_available,
            timeout=timeout,
            cancelled=cancelled,
        )


__all__ = ["NativePlanExecutionAdapter"]
