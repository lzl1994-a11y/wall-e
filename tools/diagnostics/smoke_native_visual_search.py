#!/usr/bin/env python3
"""Submit a bounded visual-search plan and verify that the native BT accepts it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rclpy
from std_msgs.msg import String

from services.behavior_tree_protocol import (
    BEHAVIOR_TREE_CANCEL_TOPIC,
    BEHAVIOR_TREE_EXECUTE_TOPIC,
    BEHAVIOR_TREE_STATUS_TOPIC,
    encode_plan_cancel,
    parse_plan_status,
)
from services.visual_search import compile_visual_search_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the native BehaviorTree.CPP node can load its visual-search tree."
    )
    parser.add_argument("--target", default="smoke-test-target")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    plan = compile_visual_search_plan(
        turn_id="native-visual-search-smoke",
        arguments={"target": args.target, "max_views": 2},
    )
    payload = json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":"))
    accepted = False
    terminal: dict[str, object] | None = None

    rclpy.init()
    node = rclpy.create_node("native_visual_search_smoke")
    publisher = node.create_publisher(String, BEHAVIOR_TREE_EXECUTE_TOPIC, 10)
    cancel_publisher = node.create_publisher(String, BEHAVIOR_TREE_CANCEL_TOPIC, 10)

    def on_status(message: String) -> None:
        nonlocal accepted, terminal
        status = parse_plan_status(message.data)
        if status is not None and status["plan_id"] == plan.plan_id:
            if status["status"] == "accepted":
                accepted = True
            elif status["status"] in {"success", "failure", "halted", "rejected"}:
                terminal = status

    subscription = node.create_subscription(
        String, BEHAVIOR_TREE_STATUS_TOPIC, on_status, 10
    )

    deadline = time.monotonic() + args.timeout
    next_publish = 0.0
    cancel_sent = False
    try:
        while terminal is None and time.monotonic() < deadline:
            now = time.monotonic()
            if not accepted and now >= next_publish and publisher.get_subscription_count() > 0:
                publisher.publish(String(data=payload))
                next_publish = now + 0.5
            if accepted and not cancel_sent and cancel_publisher.get_subscription_count() > 0:
                cancel_publisher.publish(String(data=encode_plan_cancel(plan.plan_id)))
                cancel_sent = True
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if terminal is None:
        print("native visual-search smoke: timed out waiting for terminal status")
        return 1
    print(
        "native visual-search smoke:",
        json.dumps(
            {
                "accepted": accepted,
                "status": terminal["status"],
                "error": terminal["error"],
                "source": terminal["source"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    return 0 if accepted and terminal["status"] == "halted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
