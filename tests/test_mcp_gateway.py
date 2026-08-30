import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastmcp import Client
from fastmcp.exceptions import ToolError

from services.mcp_gateway import (
    MCP_TOKEN_ENV,
    McpGatewaySettings,
    create_mcp_gateway,
    load_mcp_gateway_settings,
    require_safe_transport,
    token_from_environment,
)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments, *, timeout):
        self.calls.append((name, arguments, timeout))
        return {"status": "completed", "action": name}


class McpGatewayTests(unittest.TestCase):
    def test_network_listener_requires_environment_token(self):
        settings = McpGatewaySettings(host="0.0.0.0")
        with self.assertRaisesRegex(RuntimeError, MCP_TOKEN_ENV):
            require_safe_transport(settings, None)
        require_safe_transport(settings, "s" * 32)

    def test_short_or_non_ascii_transport_tokens_are_rejected(self):
        settings = McpGatewaySettings(host="0.0.0.0")
        with self.assertRaisesRegex(RuntimeError, "32-512"):
            require_safe_transport(settings, "short")
        with self.assertRaisesRegex(RuntimeError, "ASCII"):
            require_safe_transport(settings, "令牌" * 20)

    def test_token_is_read_only_from_environment(self):
        with patch.dict(os.environ, {MCP_TOKEN_ENV: "  token-value  "}, clear=False):
            self.assertEqual(token_from_environment(), "token-value")

    def test_config_token_is_preferred_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("mcp:\n  token: " + "c" * 32 + "\n", encoding="utf-8")
            with patch("services.mcp_gateway.DEFAULT_CONFIG_PATH", path), patch.dict(
                os.environ, {MCP_TOKEN_ENV: "e" * 32}, clear=False
            ):
                self.assertEqual(token_from_environment(), "c" * 32)

    def test_invalid_settings_fail_safe_to_local_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "mcp:\n  enabled: 1\n  host: ''\n  port: 70000\n  path: relative\n"
                "  command_timeout_sec: 0\n",
                encoding="utf-8",
            )
            settings = load_mcp_gateway_settings(path)
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 5555)
        self.assertEqual(settings.path, "/mcp")

    def test_server_exposes_only_curated_tools_and_executes_through_bridge(self):
        executor = FakeExecutor()
        server = create_mcp_gateway(executor, McpGatewaySettings())

        async def scenario():
            async with Client(server) as client:
                tools = await client.list_tools()
                result = await client.call_tool(
                    "move_chassis",
                    {"direction": "forward", "duration": 1},
                )
                return tools, result

        tools, result = asyncio.run(scenario())
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "move_chassis",
                "play_sequence",
                "express_emotion",
                "set_tracking_mode",
                "set_vision_gate",
                "stop_all",
            },
        )
        move_tool = next(tool for tool in tools if tool.name == "move_chassis")
        duration_schema = move_tool.inputSchema["properties"]["duration"]
        self.assertEqual(duration_schema["minimum"], 1)
        self.assertEqual(duration_schema["maximum"], 3)
        self.assertTrue(move_tool.annotations.destructiveHint)
        self.assertEqual(result.data["status"], "completed")
        self.assertEqual(
            executor.calls,
            [("move_chassis", {"direction": "forward", "duration": 1}, 13.0)],
        )

    def test_runtime_argument_limits_reject_out_of_range_motion(self):
        executor = FakeExecutor()
        server = create_mcp_gateway(executor, McpGatewaySettings())

        async def scenario():
            async with Client(server) as client:
                return await client.call_tool(
                    "move_chassis",
                    {"direction": "forward", "duration": 9},
                )

        with self.assertRaisesRegex(ToolError, "maximum of 3"):
            asyncio.run(scenario())
        self.assertEqual(executor.calls, [])


if __name__ == "__main__":
    unittest.main()
