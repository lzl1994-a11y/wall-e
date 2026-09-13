#!/usr/bin/env python3
"""Verify the standard ExecuteTree action bridge without moving hardware."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import rclpy
from btcpp_ros2_interfaces.action import ExecuteTree
from rclpy.action import ActionClient


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="exercise the repository's Ros2ActionPlanExecutor instead of a raw client",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("wali_bt_ros2_bridge_smoke")
    client = None
    try:
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
        if args.adapter:
            from services.ros2_action_execution import (
                Ros2ActionPlanExecutor,
                create_wali_task_action_client,
            )

            class InvalidPlan:
                def __init__(self, value, encoded_payload):
                    self.plan_id = value
                    self.steps = ()
                    self._payload = encoded_payload

                def to_dict(self):
                    return json.loads(self._payload)

            spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
            spin_thread.start()
            adapter_client, goal_type = create_wali_task_action_client(
                node, ROOT
            )
            result = Ros2ActionPlanExecutor(adapter_client, goal_type).try_execute(
                InvalidPlan(plan_id, payload), timeout=args.timeout
            )
            adapter_client.destroy()
            if result is None:
                print("BehaviorTree.ROS2 bridge smoke: adapter found no server")
                return 1
            print(
                "BehaviorTree.ROS2 bridge adapter smoke:",
                json.dumps(
                    {"status": result.get("status"), "error": result.get("error", "")},
                    separators=(",", ":"),
                ),
            )
            return 0 if result.get("status") == "rejected" else 1

        client = ActionClient(node, ExecuteTree, "wali_task")
        if not client.wait_for_server(timeout_sec=args.timeout):
            print("BehaviorTree.ROS2 bridge smoke: action server unavailable")
            return 1
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
        if client is not None:
            client.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
