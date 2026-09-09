#!/usr/bin/env python3
"""Single ingress owner for robot actions with priority/resource arbitration."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.action_arbitration import ActionArbiter
from services.action_command import (
    ACTION_COMMAND_TOPIC,
    ACTION_REQUEST_TOPIC,
    new_action_request_id,
    parse_action_request,
)
from services.action_status import (
    ACTION_STATUS_TOPIC,
    TERMINAL_ACTION_STATUSES,
    build_action_status,
    parse_action_status,
)


class ActionCoordinatorNode(Node):
    def __init__(self):
        super().__init__("action_coordinator_node")
        self._arbiter = ActionArbiter(
            lease_timeout=self.declare_parameter("lease_timeout_sec", 30.0).value
        )
        self._command_pub = self.create_publisher(String, ACTION_COMMAND_TOPIC, 20)
        self._status_pub = self.create_publisher(String, ACTION_STATUS_TOPIC, 20)
        self.create_subscription(String, ACTION_REQUEST_TOPIC, self._on_request, 20)
        self.create_subscription(String, ACTION_STATUS_TOPIC, self._on_status, 50)
        self.create_timer(1.0, self._expire_leases)
        self.get_logger().info(
            f"Action coordinator online: {ACTION_REQUEST_TOPIC} -> {ACTION_COMMAND_TOPIC}"
        )

    def _on_request(self, message):
        request = parse_action_request(message.data)
        if request is None:
            self.get_logger().warning("Rejected malformed action request")
            return
        request_id = request.get("request_id") or new_action_request_id()
        source = request.get("source") or "legacy"
        decision = self._arbiter.submit(
            request_id,
            request["name"],
            source,
        )
        if not decision.accepted:
            self._status_pub.publish(String(data=build_action_status(
                request_id,
                request["name"],
                "rejected",
                source="action_coordinator",
                detail=decision.reason,
            )))
            return

        try:
            envelope = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            envelope = {}
        envelope.update({
            "name": request["name"],
            "arguments": request["arguments"],
            "request_id": request_id,
            "source": source,
        })
        if decision.preempted:
            self.get_logger().info(
                f"{source}:{request['name']} preempted {len(decision.preempted)} request(s)"
            )
        self._command_pub.publish(String(data=json.dumps(
            envelope, ensure_ascii=False, separators=(",", ":")
        )))

    def _on_status(self, message):
        status = parse_action_status(message.data)
        if status and status["status"] in TERMINAL_ACTION_STATUSES:
            self._arbiter.release(status["request_id"])

    def _expire_leases(self):
        expired = self._arbiter.expire()
        if expired:
            self.get_logger().warning(f"Expired {len(expired)} stale action lease(s)")


def main(args=None):
    rclpy.init(args=args)
    node = ActionCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
