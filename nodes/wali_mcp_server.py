#!/usr/bin/env python3
"""Streamable HTTP MCP gateway backed by the existing ROS action pipeline."""

from __future__ import annotations

import threading
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from services.action_command import build_action_cmd, new_action_request_id
from services.action_status import (
    ACTION_STATUS_TOPIC,
    TERMINAL_ACTION_STATUSES,
    parse_action_status,
)
from services.mcp_gateway import (
    create_mcp_gateway,
    load_mcp_gateway_settings,
    require_safe_transport,
    token_from_environment,
)


class RosActionExecutor(Node):
    """Publish correlated commands and wait for the owning ROS node's status."""

    def __init__(self):
        super().__init__("wali_mcp_gateway")
        self._publisher = self.create_publisher(String, "/action_cmd", 10)
        self.create_subscription(String, ACTION_STATUS_TOPIC, self._on_status, 10)
        self._condition = threading.Condition()
        self._statuses: dict[str, dict[str, str]] = {}
        self.get_logger().info("Wali MCP ROS execution bridge online")

    def _on_status(self, message):
        status = parse_action_status(message.data)
        if status is None:
            return
        with self._condition:
            self._statuses[status["request_id"]] = status
            self._condition.notify_all()

    def _wait_for_action_owner(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() > 0:
                return True
            time.sleep(0.05)
        return False

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + timeout
        if not self._wait_for_action_owner(min(deadline, started + 2.0)):
            return {
                "status": "failed",
                "action": name,
                "reason": "ros_action_owner_unavailable",
            }

        request_id = new_action_request_id()
        payload = build_action_cmd(
            name,
            arguments,
            request_id=request_id,
            source="mcp",
        )
        with self._condition:
            self._statuses.pop(request_id, None)
        self._publisher.publish(String(data=payload))

        latest = None
        with self._condition:
            while time.monotonic() < deadline:
                latest = self._statuses.get(request_id)
                if latest and latest["status"] in TERMINAL_ACTION_STATUSES:
                    break
                self._condition.wait(timeout=max(0.01, deadline - time.monotonic()))
            latest = self._statuses.pop(request_id, latest)

        if latest is None or latest["status"] not in TERMINAL_ACTION_STATUSES:
            return {
                "status": "timeout",
                "action": name,
                "request_id": request_id,
                "reason": "no_terminal_executor_status",
                **({"last_status": latest["status"]} if latest else {}),
            }
        return {
            "status": latest["status"],
            "action": name,
            "request_id": request_id,
            "executor": latest["source"],
            **({"reason": latest["detail"]} if latest["detail"] else {}),
        }


def main(args=None):
    settings = load_mcp_gateway_settings()
    token = token_from_environment()
    require_safe_transport(settings, token)
    rclpy.init(args=args)
    executor = RosActionExecutor()
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(executor,),
        name="wali-mcp-ros-spin",
        daemon=True,
    )
    try:
        spin_thread.start()
        server = create_mcp_gateway(executor, settings, token=token)
        server.run(
            transport="http",
            host=settings.host,
            port=settings.port,
            path=settings.path,
            stateless_http=True,
        )
    finally:
        executor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
