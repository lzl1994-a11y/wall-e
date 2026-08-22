import builtins
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from services import mcp_service
from services.tool_dispatcher import parse_action_cmd


class FastMcpToolTests(unittest.TestCase):
    def test_lightweight_action_parser_imports_without_fastmcp_or_mcp_service(self):
        real_import = builtins.__import__

        def block_optional_llm_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "services.mcp_service" or name.startswith("fastmcp"):
                raise ImportError(f"blocked optional dependency: {name}")
            return real_import(name, globals, locals, fromlist, level)

        sys.modules.pop("services.action_command", None)
        with patch("builtins.__import__", side_effect=block_optional_llm_dependencies):
            action_command = importlib.import_module("services.action_command")
            parsed = action_command.parse_action_cmd(
                '{"name":"play_sequence","arguments":"{\\"sequence_name\\":\\"turn_head_left\\"}"}'
            )
        self.assertEqual(parsed, ("play_sequence", {"sequence_name": "turn_head_left"}))

    def test_fastmcp_2x_enumerates_six_openai_tools_with_schemas(self):
        tools = mcp_service.get_chat_tools()
        self.assertEqual(len(tools), 6)
        self.assertEqual(
            {item["function"]["name"] for item in tools},
            {
                "express_emotion",
                "play_sequence",
                "move_chassis",
                "set_tracking_mode",
                "set_vision_gate",
                "inspect_camera",
            },
        )
        for item in tools:
            self.assertEqual(item["type"], "function")
            function = item["function"]
            self.assertIsInstance(function["description"], str)
            self.assertTrue(function["description"])
            self.assertIsInstance(function["parameters"], dict)
            self.assertEqual(function["parameters"]["type"], "object")

    def test_empty_fastmcp_registry_is_diagnostic_error_not_silent_empty_tools(self):
        async def no_tools():
            return {}

        with patch.object(mcp_service.mcp, "get_tools", no_tools):
            with self.assertRaisesRegex(mcp_service.MCPToolDiscoveryError, "未枚举"):
                mcp_service.get_chat_tools()


class _ToolCallResponse:
    def __iter__(self):
        tool_call = types.SimpleNamespace(
            index=0,
            function=types.SimpleNamespace(
                name="play_sequence",
                arguments='{"sequence_name":"turn_head_left"}',
            ),
        )
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )]
        )


class LlmToolAvailabilityTests(unittest.TestCase):
    def _service(self):
        from services.llm_service import LLMService

        service = object.__new__(LLMService)
        service.settings = {"temperature": 0.2, "max_tokens": 128}
        service.system_prompt = "你是瓦力。"
        service.model = "glm-4.1v-thinking-flashx"
        service.client = MagicMock()
        service._tools = None
        return service

    def test_empty_tools_raise_before_request_instead_of_silent_text_fallback(self):
        from services.llm_service import ToolCallingUnavailableError

        service = self._service()
        with patch("services.llm_service.get_tools", return_value=[]):
            with self.assertRaisesRegex(ToolCallingUnavailableError, "动作工具为空"):
                list(service.chat_stream("转个头", tools_enabled=True))
        service.client.chat.completions.create.assert_not_called()

    def test_streamed_tool_call_has_action_ready_name_and_arguments(self):
        service = self._service()
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        with patch("services.llm_service.get_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("转个头", tools_enabled=True))
        self.assertIn(
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name": "turn_head_left"}'},
            events,
        )
        request = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["tool_choice"], "auto")
        self.assertIn("tools", request)

    def test_known_thinking_model_uses_default_tool_model_but_non_tools_keep_primary(self):
        service = self._service()
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        tool_schema = [{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]
        with patch("services.llm_service.get_tools", return_value=tool_schema):
            list(service.chat_stream("转个头", tools_enabled=True))
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"],
            "glm-4.6v-flash",
        )

        service.client.chat.completions.create.return_value = _ToolCallResponse()
        list(service.chat_stream("看看画面", tools_enabled=False))
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"],
            "glm-4.1v-thinking-flashx",
        )

    def test_explicit_tool_model_overrides_default_fallback(self):
        service = self._service()
        service.settings["tool_model"] = "glm-4-flash-250414"
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        with patch("services.llm_service.get_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            list(service.chat_stream("转个头", tools_enabled=True))
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"],
            "glm-4-flash-250414",
        )

    def test_tool_model_also_controls_reasoning_request_options(self):
        service = self._service()
        service.settings.update({"provider": "zhipu", "tool_model": "glm-4.7", "reasoning_effort": "fast"})
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        with patch("services.llm_service.get_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            list(service.chat_stream("转个头", tools_enabled=True))
        request = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "glm-4.7")
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})

    def test_model_tool_rejection_has_explicit_diagnostic(self):
        from services.llm_service import ToolCallingUnavailableError

        class ApiToolRejection(RuntimeError):
            status_code = 400

        service = self._service()
        service.client.chat.completions.create.side_effect = ApiToolRejection("unsupported tools parameter")
        with patch("services.llm_service.get_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            with self.assertRaisesRegex(ToolCallingUnavailableError, "function calling"):
                list(service.chat_stream("转个头", tools_enabled=True))

    def test_network_or_auth_error_is_not_misreported_as_tool_incompatibility(self):
        service = self._service()
        service.client.chat.completions.create.side_effect = RuntimeError("network timeout")
        with patch("services.llm_service.get_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            with self.assertRaises(RuntimeError) as caught:
                list(service.chat_stream("转个头", tools_enabled=True))
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(str(caught.exception), "network timeout")

    def test_llm_action_payload_is_accepted_by_sequence_dispatch_parser(self):
        # This is the exact shape llm_ros_node publishes after a streamed tool
        # event. sequence_ros_node calls the same parser before dispatch.
        action_payload = (
            '{"turn_id":"turn-head","name":"play_sequence",'
            '"arguments":"{\\"sequence_name\\":\\"turn_head_left\\"}"}'
        )
        self.assertEqual(
            parse_action_cmd(action_payload),
            ("play_sequence", {"sequence_name": "turn_head_left"}),
        )
