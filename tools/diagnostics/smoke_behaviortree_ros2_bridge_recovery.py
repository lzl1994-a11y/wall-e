#!/usr/bin/env python3
"""Exercise Action-bridge recovery without starting the robot executor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import rclpy
from btcpp_ros2_interfaces.action import ExecuteTree
from rclpy.action import ActionClient
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, default=93)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--bridge",
        type=Path,
        default=(
            ROOT
            / "install"
            / "wali_bt_ros2_bridge"
            / "lib"
            / "wali_bt_ros2_bridge"
            / "wali_bt_ros2_bridge"
        ),
    )
    return parser.parse_args()


def wait_future(future, timeout: float):
    completed = threading.Event()
    future.add_done_callback(lambda _future: completed.set())
    if not completed.wait(timeout):
        raise TimeoutError("ROS future did not complete")
    return future.result()


def goal(plan_id: str) -> ExecuteTree.Goal:
    value = ExecuteTree.Goal()
    value.target_tree = "WaliTask"
    value.payload = json.dumps(
        {
            "schema_version": 2,
            "plan_id": plan_id,
            "turn_id": "bridge-recovery-smoke",
            "root_type": "Sequence",
            "steps": [{
                "step_id": "step-01",
                "name": "play_sequence",
                "arguments": {"sequence_name": "wave_hello"},
                "grounding": "",
                "depends_on": [],
                "resources": ["servo_motion"],
                "timeout_ms": 20000,
                "max_attempts": 1,
            }],
            "on_failure": "stop_remaining",
        },
        separators=(",", ":"),
    )
    return value


def main() -> int:
    args = parse_args()
    bridge = args.bridge.resolve()
    if not bridge.is_file():
        raise FileNotFoundError(bridge)

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    rclpy.init()
    node = rclpy.create_node("wali_bt_bridge_recovery_smoke")
    status_pub = node.create_publisher(String, "/behavior_tree/status", 10)
    executions: list[str] = []
    safety_stops: list[dict] = []
    first_plan_id = "bridge-recovery-first"
    second_plan_id = "bridge-recovery-second"

    def on_execute(message: String) -> None:
        payload = json.loads(message.data)
        plan_id = payload.get("plan_id", "")
        executions.append(plan_id)
        if plan_id != second_plan_id:
            return
        # A late terminal from the recovered goal must not finish the new goal.
        status_pub.publish(String(data=json.dumps({
            "plan_id": first_plan_id,
            "status": "success",
            "results": [],
        }, separators=(",", ":"))))
        status_pub.publish(String(data=json.dumps({
            "plan_id": second_plan_id,
            "status": "success",
            "results": [],
        }, separators=(",", ":"))))

    def on_action_request(message: String) -> None:
        payload = json.loads(message.data)
        if payload.get("name") == "stop_all":
            safety_stops.append(payload)

    node.create_subscription(String, "/behavior_tree/execute", on_execute, 10)
    node.create_subscription(String, "/action_request", on_action_request, 20)
    client = ActionClient(node, ExecuteTree, "wali_task")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    environment = os.environ.copy()
    process = subprocess.Popen(
        [
            str(bridge),
            "--ros-args",
            "-p", "cancel_grace_sec:=0.5",
            "-p", "backend_loss_grace_sec:=5.0",
            "-p", "goal_timeout_sec:=10.0",
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        spin_thread.start()
        if not client.wait_for_server(timeout_sec=args.timeout):
            raise TimeoutError("Action bridge did not start")
        time.sleep(0.5)

        first_handle = wait_future(client.send_goal_async(goal(first_plan_id)), args.timeout)
        if first_handle is None or not first_handle.accepted:
            raise RuntimeError("first goal was not accepted")
        first_result_future = first_handle.get_result_async()
        wait_future(first_handle.cancel_goal_async(), args.timeout)
        first_wrapped = wait_future(first_result_future, args.timeout)
        first_result = json.loads(first_wrapped.result.return_message)

        second_handle = wait_future(
            client.send_goal_async(goal(second_plan_id)), args.timeout
        )
        if second_handle is None or not second_handle.accepted:
            raise RuntimeError("second goal was not accepted after recovery")
        second_wrapped = wait_future(second_handle.get_result_async(), args.timeout)
        second_result = json.loads(second_wrapped.result.return_message)

        passed = (
            executions[:2] == [first_plan_id, second_plan_id]
            and first_result.get("status") == "halted"
            and first_result.get("error") == "legacy_behavior_tree_cancel_timeout"
            and any(
                item.get("source") == "wali_task_action_bridge_safety"
                and item.get("plan_id") == first_plan_id
                for item in safety_stops
            )
            and second_result.get("plan_id") == second_plan_id
            and second_result.get("status") == "success"
        )
        print(json.dumps({
            "passed": passed,
            "first_status": first_result.get("status"),
            "first_error": first_result.get("error"),
            "safety_stop_count": len(safety_stops),
            "second_status": second_result.get("status"),
        }, separators=(",", ":")))
        return 0 if passed else 1
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
