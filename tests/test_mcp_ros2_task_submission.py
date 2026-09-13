"""Unit coverage for MCP's standard Action submission adapter without ROS."""

from __future__ import annotations

import importlib
import sys
import types


def _load_server_module(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    node_module = types.ModuleType("rclpy.node")
    node_module.Node = object
    rclpy.node = node_module
    std_msgs = types.ModuleType("std_msgs")
    msg_module = types.ModuleType("std_msgs.msg")
    msg_module.String = type("String", (), {})
    std_msgs.msg = msg_module
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", msg_module)
    sys.modules.pop("nodes.wali_mcp_server", None)
    return importlib.import_module("nodes.wali_mcp_server")


class _PlanExecutor:
    def __init__(self, result):
        self.result = result
        self.plan = None

    def try_execute(self, plan, *, timeout):
        self.plan = plan
        return self.result


def test_mcp_maps_standard_action_result_to_existing_tool_shape(monkeypatch):
    module = _load_server_module(monkeypatch)
    executor = object.__new__(module.RosActionExecutor)
    action_executor = _PlanExecutor({"status": "success", "source": "native_behavior_tree"})
    executor._ros2_task_executor = action_executor

    result = executor._try_execute_wali_task(
        "request-1",
        "play_sequence",
        {"sequence_name": "wave_hello"},
        timeout=5.0,
    )

    assert result["status"] == "completed"
    assert result["request_id"] == "request-1"
    assert result["executor"] == "native_behavior_tree"
    assert action_executor.plan.steps[0].name == "play_sequence"


def test_mcp_uses_legacy_fallback_only_when_standard_server_is_unavailable(monkeypatch):
    module = _load_server_module(monkeypatch)
    executor = object.__new__(module.RosActionExecutor)
    executor._ros2_task_executor = _PlanExecutor(None)

    assert executor._try_execute_wali_task(
        "request-2",
        "play_sequence",
        {"sequence_name": "wave_hello"},
        timeout=5.0,
    ) is None
