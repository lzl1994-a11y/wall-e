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
            self.assertFalse(function["parameters"]["additionalProperties"])

        by_name = {item["function"]["name"]: item["function"] for item in tools}
        self.assertEqual(
            set(by_name["move_chassis"]["parameters"]["properties"]["direction"]["enum"]),
            {"forward", "backward", "spin", "left", "right"},
        )
        self.assertEqual(
            set(by_name["set_tracking_mode"]["parameters"]["properties"]["mode"]["enum"]),
            {"follow_me", "look_at_me", "idle"},
        )

    def test_empty_fastmcp_registry_is_diagnostic_error_not_silent_empty_tools(self):
        async def no_tools():
            return {}

        with patch.object(mcp_service.mcp, "get_tools", no_tools):
            with self.assertRaisesRegex(mcp_service.MCPToolDiscoveryError, "未枚举"):
                mcp_service.get_chat_tools()

    def test_dispatcher_separates_structured_answer_from_real_action_tools(self):
        from services import tool_dispatcher

        action = {
            "type": "function",
            "function": {
                "name": "play_sequence",
                "description": "x",
                "parameters": {"type": "object"},
            },
        }
        with patch.object(tool_dispatcher.mcp, "get_chat_tools", return_value=[action]):
            structured_tools = tool_dispatcher.get_tools()
            action_tools = tool_dispatcher.get_action_tools()
        self.assertEqual(
            [tool["function"]["name"] for tool in structured_tools],
            ["direct_answer", "play_sequence"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in action_tools],
            ["play_sequence"],
        )

    def test_multimodal_direct_answer_requires_transcript_and_response(self):
        from services import tool_dispatcher

        with patch.object(tool_dispatcher.mcp, "get_chat_tools", return_value=[]):
            tools = tool_dispatcher.get_multimodal_tools()
        parameters = tools[0]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["heard_text", "response"])


class _ToolCallResponse:
    def __iter__(self):
        tool_call = [
            types.SimpleNamespace(
                index=0,
                function=types.SimpleNamespace(
                    name="play_sequence",
                    arguments='{"sequence_name":"turn_head_left"}',
                ),
            ),
        ]
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content="好的，我来转头。", tool_calls=tool_call),
                finish_reason="tool_calls",
            )]
        )


class _DirectAnswerResponse:
    def __iter__(self):
        tool_call = types.SimpleNamespace(
            index=0,
            function=types.SimpleNamespace(
                name="direct_answer",
                arguments='{"response":"看起来是一只杯子。"}',
            ),
        )
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )]
        )


class ToolCallAccumulatorTests(unittest.TestCase):
    @staticmethod
    def _delta(calls):
        return types.SimpleNamespace(tool_calls=calls)

    @staticmethod
    def _call(index, name, arguments):
        return types.SimpleNamespace(
            index=index,
            function=types.SimpleNamespace(name=name, arguments=arguments),
        )

    def test_malformed_arguments_are_discarded_instead_of_becoming_empty_object(self):
        from services.tool_dispatcher import ToolCallAccumulator

        accumulator = ToolCallAccumulator()
        accumulator.feed(self._delta([
            self._call(0, "move_chassis", '{"direction":'),
        ]))
        self.assertEqual(accumulator.flush(), [])

    def test_provider_call_order_is_preserved(self):
        from services.tool_dispatcher import ToolCallAccumulator

        accumulator = ToolCallAccumulator()
        accumulator.feed(self._delta([
            self._call(0, "play_sequence", '{"sequence_name":"wave_hello"}'),
            self._call(1, "express_emotion", '{"emotion":"happy"}'),
        ]))
        self.assertEqual(
            [call["name"] for call in accumulator.flush()],
            ["play_sequence", "express_emotion"],
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
        with patch("services.llm_service.get_action_tools", return_value=[]):
            with self.assertRaisesRegex(ToolCallingUnavailableError, "动作工具为空"):
                list(service.chat_stream("转个头", tools_enabled=True))
        service.client.chat.completions.create.assert_not_called()

    def test_tool_branch_discards_mixed_content_and_emits_action(self):
        service = self._service()
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("转个头", tools_enabled=True))
        self.assertIn(
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name": "turn_head_left"}'},
            events,
        )
        self.assertFalse(any(event["type"] == "text" for event in events))
        request = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["tool_choice"], "auto")
        self.assertIn("tools", request)

    def test_action_only_response_is_valid_without_direct_answer(self):
        class ActionOnlyResponse:
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

        service = self._service()
        service.client.chat.completions.create.return_value = ActionOnlyResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("转个头", tools_enabled=True))
        self.assertEqual(
            events[0],
            {"type": "tool_call", "name": "play_sequence", "arguments": '{"sequence_name": "turn_head_left"}'},
        )
        self.assertEqual(events[-1], {"type": "done", "finish_reason": "tool_calls"})

    def test_plain_content_is_valid_with_action_tools_enabled(self):
        class PlainResponse:
            def __iter__(self):
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content="你好，我在。", tool_calls=None),
                        finish_reason="stop",
                    )]
                )

        service = self._service()
        service.client.chat.completions.create.return_value = PlainResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("你好", tools_enabled=True))
        self.assertEqual(events[0], {"type": "text", "content": "你好，我在。"})
        self.assertEqual(events[-1], {"type": "done", "finish_reason": "stop"})

    def test_tool_call_wins_even_when_provider_streams_text_first(self):
        class TextThenToolResponse:
            def __iter__(self):
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content="我可以转头呀。", tool_calls=None),
                        finish_reason=None,
                    )]
                )
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=None, tool_calls=[
                            types.SimpleNamespace(
                                index=0,
                                function=types.SimpleNamespace(
                                    name="play_sequence",
                                    arguments='{"sequence_name":"turn_head_left"}',
                                ),
                            ),
                        ]),
                        finish_reason="tool_calls",
                    )]
                )

        service = self._service()
        service.client.chat.completions.create.return_value = TextThenToolResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("你能转头吗？", tools_enabled=True))
        self.assertFalse(any(event["type"] == "text" for event in events))
        self.assertEqual(
            events[0],
            {
                "type": "tool_call",
                "name": "play_sequence",
                "arguments": '{"sequence_name": "turn_head_left"}',
            },
        )

    def test_tool_call_wins_after_multiple_prior_text_chunks(self):
        class MultiTextThenToolResponse:
            def __iter__(self):
                for content in ("好的。", "我来转头。"):
                    yield types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=content, tool_calls=None),
                            finish_reason=None,
                        )]
                    )
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=None, tool_calls=[
                            types.SimpleNamespace(
                                index=0,
                                function=types.SimpleNamespace(
                                    name="play_sequence",
                                    arguments='{"sequence_name":"turn_head_left"}',
                                ),
                            ),
                        ]),
                        finish_reason="tool_calls",
                    )]
                )

        service = self._service()
        service.client.chat.completions.create.return_value = MultiTextThenToolResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            events = list(service.chat_stream("向左转头", tools_enabled=True))
        self.assertFalse(any(event["type"] == "text" for event in events))
        self.assertTrue(any(event["type"] == "tool_call" for event in events))

    def test_structured_request_does_not_poison_action_tool_cache(self):
        service = self._service()
        service.client.chat.completions.create.side_effect = [
            _DirectAnswerResponse(),
            _ToolCallResponse(),
        ]
        visual_events = list(service.chat_stream(
            "看图",
            tools_enabled=False,
            structured_answer=True,
        ))
        self.assertEqual(visual_events[0]["content"], "看起来是一只杯子。")
        self.assertIsNone(service._tools)

        action_schema = [{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]
        with patch("services.llm_service.get_action_tools", return_value=action_schema):
            list(service.chat_stream("转个头", tools_enabled=True))
        second_request = service.client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(second_request["tools"], action_schema)

    def test_structured_answer_still_rejects_plain_content(self):
        from services.llm_service import StructuredAnswerUnavailableError

        class PlainResponse:
            def __iter__(self):
                yield types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content="untrusted", tool_calls=None),
                        finish_reason="stop",
                    )]
                )

        service = self._service()
        service.client.chat.completions.create.return_value = PlainResponse()
        with self.assertRaisesRegex(StructuredAnswerUnavailableError, "direct_answer.response"):
            list(service.chat_stream(
                "看看画面",
                tools_enabled=False,
                structured_answer=True,
            ))
        request = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["tool_choice"],
            {"type": "function", "function": {"name": "direct_answer"}},
        )

    def test_primary_model_is_used_when_no_tool_model_is_configured(self):
        service = self._service()
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        tool_schema = [{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]
        with patch("services.llm_service.get_action_tools", return_value=tool_schema):
            list(service.chat_stream("转个头", tools_enabled=True))
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"],
            "glm-4.1v-thinking-flashx",
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
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            list(service.chat_stream("转个头", tools_enabled=True))
        self.assertEqual(
            service.client.chat.completions.create.call_args.kwargs["model"],
            "glm-4-flash-250414",
        )

    def test_structured_answer_uses_explicit_tool_model(self):
        service = self._service()
        service.settings["tool_model"] = "ernie-5.0"
        service.client.chat.completions.create.return_value = _DirectAnswerResponse()

        events = list(service.chat_stream(
            "看看画面",
            tools_enabled=False,
            structured_answer=True,
        ))

        request = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "ernie-5.0")
        self.assertEqual(events[0]["content"], "看起来是一只杯子。")

    def test_tool_model_also_controls_reasoning_request_options(self):
        service = self._service()
        service.settings.update({"provider": "zhipu", "tool_model": "glm-4.7", "reasoning_effort": "fast"})
        service.client.chat.completions.create.return_value = _ToolCallResponse()
        with patch("services.llm_service.get_action_tools", return_value=[{
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
        with patch("services.llm_service.get_action_tools", return_value=[{
            "type": "function",
            "function": {"name": "play_sequence", "description": "x", "parameters": {"type": "object"}},
        }]):
            with self.assertRaisesRegex(ToolCallingUnavailableError, "function calling"):
                list(service.chat_stream("转个头", tools_enabled=True))

    def test_network_or_auth_error_is_not_misreported_as_tool_incompatibility(self):
        service = self._service()
        service.client.chat.completions.create.side_effect = RuntimeError("network timeout")
        with patch("services.llm_service.get_action_tools", return_value=[{
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
