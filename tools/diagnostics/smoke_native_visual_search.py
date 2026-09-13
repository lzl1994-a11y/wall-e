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
    BEHAVIOR_TREE_EXECUTE_TOPIC,
    BEHAVIOR_TREE_STATUS_TOPIC,
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
    received: dict[str, object] | None = None

    rclpy.init()
    node = rclpy.create_node("native_visual_search_smoke")
    publisher = node.create_publisher(String, BEHAVIOR_TREE_EXECUTE_TOPIC, 10)

    def on_status(message: String) -> None:
        nonlocal received
        status = parse_plan_status(message.data)
        if status is not None and status["plan_id"] == plan.plan_id:
            received = status

    subscription = node.create_subscription(
        String, BEHAVIOR_TREE_STATUS_TOPIC, on_status, 10
    )
    del subscription

    deadline = time.monotonic() + args.timeout
    next_publish = 0.0
    try:
        while received is None and time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish and publisher.get_subscription_count() > 0:
                publisher.publish(String(data=payload))
                next_publish = now + 0.5
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if received is None:
        print("native visual-search smoke: timed out waiting for plan status")
        return 1
    print(
        "native visual-search smoke:",
        json.dumps(
            {
                "status": received["status"],
                "error": received["error"],
                "source": received["source"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    return 0 if received["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
