"""Pure priority and resource arbitration for the unified action ingress."""

from __future__ import annotations

import time
from dataclasses import dataclass

from services.action_registry import get_action_skill


def source_priority(source: str, action_name: str) -> int:
    source = str(source or "legacy").lower()
    if action_name == "stop_all" or "cancel" in source or "safety" in source:
        return 1000
    if "joystick" in source or source == "joy":
        return 100
    if source == "mcp" or source.startswith("mcp_"):
        return 70
    if "behavior_tree" in source:
        return 60
    if "dialog" in source or source.startswith("llm") or source.startswith("voice"):
        return 50
    return 40


@dataclass(frozen=True)
class ActionLease:
    request_id: str
    name: str
    source: str
    priority: int
    resources: frozenset[str]
    owner: str
    supports_cancel: bool
    expires_at: float


@dataclass(frozen=True)
class ArbitrationDecision:
    accepted: bool
    reason: str = ""
    preempted: tuple[ActionLease, ...] = ()


class ActionArbiter:
    """Allow independent owners concurrently and serialize each owner safely."""

    def __init__(self, *, lease_timeout: float = 30.0):
        self._lease_timeout = max(1.0, float(lease_timeout))
        self._leases: dict[str, ActionLease] = {}

    @property
    def active_leases(self) -> tuple[ActionLease, ...]:
        return tuple(self._leases.values())

    def submit(
        self,
        request_id: str,
        name: str,
        source: str,
        *,
        now: float | None = None,
    ) -> ArbitrationDecision:
        timestamp = time.monotonic() if now is None else float(now)
        self.expire(now=timestamp)
        skill = get_action_skill(name)
        if skill is None or not skill.action_bus:
            return ArbitrationDecision(False, "unknown_action")
        resources = skill.arbitration_resources
        if request_id in self._leases:
            return ArbitrationDecision(False, "duplicate_request_id")

        priority = source_priority(source, name)
        conflicts = [
            lease
            for lease in self._leases.values()
            if "*" in resources
            or "*" in lease.resources
            or not resources.isdisjoint(lease.resources)
        ]
        blocking = [lease for lease in conflicts if lease.priority > priority]
        if blocking:
            owner = max(blocking, key=lambda lease: lease.priority)
            return ArbitrationDecision(
                False,
                f"resource_busy:{owner.source}:{owner.name}",
            )

        preempted = tuple(conflicts)
        for lease in preempted:
            self._leases.pop(lease.request_id, None)
        self._leases[request_id] = ActionLease(
            request_id=request_id,
            name=name,
            source=source,
            priority=priority,
            resources=resources,
            owner=skill.owner,
            supports_cancel=skill.supports_cancel,
            expires_at=timestamp + self._lease_timeout,
        )
        return ArbitrationDecision(True, preempted=preempted)

    def release(self, request_id: str) -> bool:
        return self._leases.pop(request_id, None) is not None

    def expire(self, *, now: float | None = None) -> tuple[ActionLease, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        expired = tuple(
            lease for lease in self._leases.values() if lease.expires_at <= timestamp
        )
        for lease in expired:
            self._leases.pop(lease.request_id, None)
        return expired


__all__ = [
    "ActionArbiter",
    "ActionLease",
    "ArbitrationDecision",
    "source_priority",
]
