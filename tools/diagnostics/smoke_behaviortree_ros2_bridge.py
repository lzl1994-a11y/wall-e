#!/usr/bin/env python3
"""Verify the standard ExecuteTree action bridge without moving hardware."""

from __future__ import annotations

import argparse
import json

import rclpy
from btcpp_ros2_interfaces.action import ExecuteTree
from rclpy.action import ActionClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("wali_bt_ros2_bridge_smoke")
    client = ActionClient(node, ExecuteTree, "wali_task")
    try:
        if not client.wait_for_server(timeout_sec=args.timeout):
            print("BehaviorTree.ROS2 bridge smoke: action server unavailable")
            return 1

        # Deliberately invalid schema: the legacy executor must reject it before
        # any action leaf or hardware command can run. This still verifies the
        # standard Action goal, forwarding, terminal result, and correlation.
        plan_id = "ros2-bridge-smoke"
        payload = json.dumps(
            {
                "schema_version": 999,
                "plan_id": plan_id,
                "turn_id": "ros2-bridge-smoke",
                "root_type": "Sequence",
                "steps": [],
                "on_failure": "stop_remaining",
            },
            separators=(",", ":"),
        )
        goal = ExecuteTree.Goal()
        goal.target_tree = "WaliTask"
        goal.payload = payload
        goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, goal_future, timeout_sec=args.timeout)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("BehaviorTree.ROS2 bridge smoke: goal was not accepted")
            return 1

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=args.timeout)
        wrapped = result_future.result()
        if wrapped is None:
            print("BehaviorTree.ROS2 bridge smoke: result timed out")
            return 1
        try:
            result_payload = json.loads(wrapped.result.return_message)
        except (TypeError, json.JSONDecodeError):
            print("BehaviorTree.ROS2 bridge smoke: terminal payload was not JSON")
            return 1
        passed = (
            result_payload.get("plan_id") == plan_id
            and result_payload.get("status") == "rejected"
        )
        print(
            "BehaviorTree.ROS2 bridge smoke:",
            json.dumps(
                {
                    "goal_accepted": True,
                    "legacy_status": result_payload.get("status"),
                    "error": result_payload.get("error", ""),
                },
                separators=(",", ":"),
            ),
        )
        return 0 if passed else 1
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
