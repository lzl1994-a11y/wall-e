"""Validated single source of truth for high-level robot action skills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


ACTION_SKILL_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "core" / "action_skills.json"
)


@dataclass(frozen=True)
class ActionSkill:
    name: str
    owner: str
    plan_resources: tuple[str, ...]
    arbitration_resources: frozenset[str]
    action_bus: bool
    timeout_ms: int
    max_attempts: int
    supports_cancel: bool


@lru_cache(maxsize=1)
def load_action_skills() -> dict[str, ActionSkill]:
    data = json.loads(ACTION_SKILL_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported action skill registry schema")
    raw_skills = data.get("skills")
    if not isinstance(raw_skills, dict) or not raw_skills:
        raise ValueError("action skill registry must contain skills")

    skills: dict[str, ActionSkill] = {}
    for name, raw in raw_skills.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("invalid action skill entry")
        owner = raw.get("owner")
        plan_resources = raw.get("plan_resources")
        arbitration_resources = raw.get("arbitration_resources")
        action_bus = raw.get("action_bus")
        timeout_ms = raw.get("timeout_ms")
        max_attempts = raw.get("max_attempts")
        supports_cancel = raw.get("supports_cancel")
        if (
            not isinstance(owner, str)
            or not owner
            or not isinstance(plan_resources, list)
            or not plan_resources
            or any(not isinstance(item, str) or not item for item in plan_resources)
            or not isinstance(arbitration_resources, list)
            or not arbitration_resources
            or any(
                not isinstance(item, str) or not item
                for item in arbitration_resources
            )
            or not isinstance(action_bus, bool)
            or not isinstance(timeout_ms, int)
            or not 100 <= timeout_ms <= 60_000
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 3
            or not isinstance(supports_cancel, bool)
        ):
            raise ValueError(f"invalid action skill definition: {name}")
        skills[name] = ActionSkill(
            name=name,
            owner=owner,
            plan_resources=tuple(plan_resources),
            arbitration_resources=frozenset(arbitration_resources),
            action_bus=action_bus,
            timeout_ms=timeout_ms,
            max_attempts=max_attempts,
            supports_cancel=supports_cancel,
        )
    return skills


def get_action_skill(name: str) -> ActionSkill | None:
    return load_action_skills().get(name)


__all__ = [
    "ACTION_SKILL_REGISTRY_PATH",
    "ActionSkill",
    "get_action_skill",
    "load_action_skills",
]
