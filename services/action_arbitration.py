"""Pure priority and resource arbitration for the unified action ingress."""

from __future__ import annotations

import time
from dataclasses import dataclass


_ACTION_RESOURCES: dict[str, frozenset[str]] = {
    # These actions share sequence_ros_node, whose current contract is one
    # active motion at a time even when chassis and servo hardware differ.
    "express_emotion": frozenset({"motion_owner"}),
    "move_chassis": frozenset({"motion_owner"}),
    "manual_servo": frozenset({"motion_owner"}),
    "play_sequence": frozenset({"motion_owner"}),
    "control_music": frozenset({"music_owner"}),
    "set_tracking_mode": frozenset({"tracking_owner"}),
    "set_vision_gate": frozenset({"tracking_owner"}),
    "stop_all": frozenset({"*"}),
}


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
    expires_at: float


@dataclass(frozen=True)
class ArbitrationDecision:
    accepted: bool
    reason: str = ""
    preempted: tuple[str, ...] = ()


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
        resources = _ACTION_RESOURCES.get(name)
        if resources is None:
            return ArbitrationDecision(False, "unknown_action")
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

        preempted = tuple(lease.request_id for lease in conflicts)
        for old_request_id in preempted:
            self._leases.pop(old_request_id, None)
        self._leases[request_id] = ActionLease(
            request_id=request_id,
            name=name,
            source=source,
            priority=priority,
            resources=resources,
            expires_at=timestamp + self._lease_timeout,
        )
        return ArbitrationDecision(True, preempted=preempted)

    def release(self, request_id: str) -> bool:
        return self._leases.pop(request_id, None) is not None

    def expire(self, *, now: float | None = None) -> tuple[str, ...]:
        timestamp = time.monotonic() if now is None else float(now)
        expired = tuple(
            request_id
            for request_id, lease in self._leases.items()
            if lease.expires_at <= timestamp
        )
        for request_id in expired:
            self._leases.pop(request_id, None)
        return expired


__all__ = [
    "ActionArbiter",
    "ActionLease",
    "ArbitrationDecision",
    "source_priority",
]
