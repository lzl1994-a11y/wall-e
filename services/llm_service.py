# services/llm_service.py
import json
import logging
import yaml
from openai import OpenAI
from services.llm_output_filter import VisibleAnswerFilter
from services.llm_prompt import (
    with_action_tool_policy,
    with_dialog_expression_policy,
    with_direct_speech_policy,
    with_structured_answer_policy,
)
from services.llm_request_options import reasoning_request_options
from services.tool_dispatcher import (
    DIRECT_ANSWER_TOOL,
    DIRECT_ANSWER_TOOL_NAME,
    ToolCallAccumulator,
    get_action_tools,
)
from services.dialog_expression_protocol import normalize_expression


LOGGER = logging.getLogger(__name__)

class ToolCallingUnavailableError(RuntimeError):
    """The configured model or MCP registry cannot service a tools-enabled turn."""


class StructuredAnswerUnavailableError(RuntimeError):
    """A structured-only request did not return the required direct_answer."""


def _text_only_chat_history(chat_history):
    """Copy history without carrying prior image/audio payloads forward.

    A current image may still be attached explicitly by ``chat_stream`` below,
    but data URLs from earlier turns never become model context or retained
    history through this service boundary.
    """
    cleaned = []
    for item in chat_history or []:
        if not isinstance(item, dict):
            continue
        message = dict(item)
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                value = block.get("text")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            message["content"] = "\n".join(text_parts)
        elif content is not None and not isinstance(content, str):
            message["content"] = ""
        cleaned.append(message)
    return cleaned


def _is_tool_calling_rejection(exc: Exception) -> bool:
    """Identify only explicit provider rejections of tools/function parameters."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", "")
    detail = f"{exc} {body}".lower()
    mentions_tools = "tool" in detail or "function" in detail
    rejection = any(marker in detail for marker in (
        "unsupported", "not support", "not supported", "unknown parameter",
        "invalid parameter", "unrecognized parameter", "not available",
    ))
    return mentions_tools and rejection and status_code in (400, 404, 422)


class LLMService:
    def __init__(self, config_path="core/config.yaml"):
        # [ZH] 1. 从配置文件读取配置
        # [EN] 1. Load configuration from YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.settings = config['llm']
        self.system_prompt = config['system_prompt']
        
        # [ZH] 2. 初始化 OpenAI 客户端 (在此项目中指向 Kimi 的接口)
        # [EN] 2. Initialize OpenAI client (pointing to Kimi API in this project)
        self.client = OpenAI(
            api_key=self.settings['key'],
            base_url=self.settings['url']
        )
        self.model = self.settings['model']
        self.tool_model, self.tool_model_source = self._resolve_tool_model()
        if self.tool_model != self.model:
            LOGGER.warning(
                "LLM tools-enabled requests use %s (%s); tools-disabled visual/retry requests remain on %s",
                self.tool_model,
                self.tool_model_source,
                self.model,
            )
        self._tools = None

    def _resolve_tool_model(self):
        configured = self.settings.get("tool_model") if isinstance(self.settings, dict) else None
        if isinstance(configured, str) and configured.strip():
            return configured.strip(), "llm.tool_model"
        return self.model, "primary model"

    def _request_model(self, tools_enabled):
        if not tools_enabled:
            return self.model
        # Tests and compatibility callers may construct this class with
        # __new__; resolve lazily if __init__ did not set the cached value.
        return getattr(self, "tool_model", self._resolve_tool_model()[0])

    @staticmethod
    def _json_fallback_actions(value, action_tools):
        """Keep only well-formed calls to tools offered in this request."""
        offered = {
            tool["function"]["name"]
            for tool in action_tools or []
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        }
        raw_actions = value.get("actions", [])
        if not isinstance(raw_actions, list):
            return []
        actions = []
        for item in raw_actions[:3]:
            if not isinstance(item, dict):
                continue
            if set(item) == {"name", "arguments"}:
                name = item.get("name")
                arguments = item.get("arguments")
            elif set(item) == {"action", "parameters"}:
                # Baidu JSON Object mode currently prefers these aliases even
                # when the requested schema says name/arguments.
                name = item.get("action")
                arguments = item.get("parameters")
            else:
                continue
            if name not in offered or not isinstance(arguments, dict):
                continue
            actions.append({"name": name, "arguments": arguments})
        return actions

    @staticmethod
    def _parse_json_object(content):
        """Parse a provider JSON response, tolerating one outer code fence."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("JSON dialog answer is empty")
        candidate = content.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(candidate[start:end + 1])
        if not isinstance(value, dict):
            raise ValueError("JSON dialog answer is not an object")
        return value

    @staticmethod
    def _fallback_intensity(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = max(0.0, min(1.0, float(value)))
            return "low" if value < 0.34 else "medium" if value < 0.67 else "high"
        return value

    def _json_dialog_answer(self, messages, model, max_tokens, action_tools=None):
        """Provider fallback when an advertised forced tool call is ignored."""
        fallback_messages = [dict(message) for message in messages]
        action_tools = list(action_tools or [])
        action_catalog = [
            {
                "name": tool["function"].get("name"),
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {}),
            }
            for tool in action_tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ]
        fallback_messages[0]["content"] = (
            str(fallback_messages[0].get("content", ""))
            + "\n\n只返回一个 JSON 对象，不要 Markdown 或其他文字。必须包含 response、"
              "expression、intensity。expression 只能是 neutral、listening、thinking、"
              "happy、sad、surprised、confused、concerned；intensity 只能是 low、"
              "medium、high。还必须包含 actions 数组。只有用户明确命令瓦力现在执行现实"
              "动作时，actions 才能填写；普通聊天、能力询问、否定、引用、故事或第三方"
              "动作必须返回空数组。每项严格使用 {\"name\":工具名,\"arguments\":参数对象}。"
              "这是 Function Calling 失效后的 JSON 回退，本次必须把应执行的动作写入 actions，"
              "不能只在 response 中声称已经执行。可用动作工具如下：\n"
            + json.dumps(action_catalog, ensure_ascii=False, separators=(",", ":"))
        )
        kwargs = {
            "model": model,
            "messages": fallback_messages,
            "temperature": self.settings.get("temperature", 0.3),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request_settings = (
            self.settings
            if model == self.model
            else {**self.settings, "model": model}
        )
        kwargs.update(reasoning_request_options(request_settings))
        result = self.client.chat.completions.create(**kwargs)
        content = result.choices[0].message.content
        value = self._parse_json_object(content)
        response = value.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("JSON dialog answer has no response")
        expression, intensity = normalize_expression(
            value.get("expression"), self._fallback_intensity(value.get("intensity"))
        )
        actions = self._json_fallback_actions(value, action_tools)
        return response.strip(), expression, intensity, actions

    def chat_stream(
        self,
        user_text,
        chat_history=None,
        image_base64=None,
        tools_enabled=True,
        structured_answer=False,
        system_prompt=None,
        max_tokens_override=None,
    ):
        """
        [ZH] 发起流式对话 (Generator)。
             yield 两种数据: "text" (供 TTS 播报) 和 "tool_call" (供硬件执行)。
        [EN] Initiate streaming chat (Generator).
             Yields two types of data: "text" (for TTS) and "tool_call" (for hardware execution).
        """
        if chat_history is None:
            chat_history = []

        # [ZH] 构建消息上下文
        # [EN] Build message context
        selected_system_prompt = (
            self.system_prompt if system_prompt is None else system_prompt
        )
        if tools_enabled and structured_answer:
            raise ValueError(
                "structured_answer is reserved for direct-answer-only requests; "
                "disable action tools for this request"
            )
        requires_structured_answer = bool(structured_answer)
        system_content = with_direct_speech_policy(selected_system_prompt)
        if tools_enabled:
            system_content = with_action_tool_policy(system_content)
            system_content = with_dialog_expression_policy(system_content)
        if requires_structured_answer:
            system_content = with_structured_answer_policy(system_content)
        messages = [{
            "role": "system",
            "content": system_content,
        }]
        messages.extend(_text_only_chat_history(chat_history))
        if image_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_text})

        # [ZH] 发起长连接流式请求
        # [EN] Send long-connection streaming request
        # Structured direct answers are implemented with a forced function
        # call too.  Use the configured tool-capable model for those requests
        # so a visual primary model without Function Calling can still produce
        # the project's fail-closed direct_answer contract.
        request_model = self._request_model(
            tools_enabled or requires_structured_answer
        )
        request_kwargs = {
            "model": request_model,
            "messages": messages,
            "temperature": self.settings.get('temperature', 0.3),
            "max_tokens": (
                max_tokens_override
                if max_tokens_override is not None
                else self.settings.get('max_tokens', 1024)
            ),
            "stream": True,
        }
        tools = []
        if tools_enabled:
            if getattr(self, "_tools", None) is None:
                action_tools = get_action_tools()
                if not action_tools:
                    raise ToolCallingUnavailableError(
                        "动作工具为空；拒绝以无工具模式发送请求。请检查 FastMCP 2.x 工具注册。"
                    )
                self._tools = [DIRECT_ANSWER_TOOL, *action_tools]
            tools = self._tools
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        elif requires_structured_answer:
            tools = [DIRECT_ANSWER_TOOL]
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": DIRECT_ANSWER_TOOL_NAME},
            }
        request_settings = (
            self.settings
            if request_model == self.model
            else {**self.settings, "model": request_model}
        )
        request_kwargs.update(reasoning_request_options(request_settings))
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if tools and _is_tool_calling_rejection(exc):
                raise ToolCallingUnavailableError(
                    f"模型 {request_model!r} 拒绝或不支持 function calling；"
                    "请检查模型能力和兼容 API 的 tools 支持。"
                ) from exc
            raise

        acc = ToolCallAccumulator()
        answer_filter = VisibleAnswerFilter()
        finish_reason = ""
        pending_tool_text = []
        tool_call_seen = False

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = str(chunk_finish_reason)
            delta = choice.delta

            if getattr(delta, "tool_calls", None):
                tool_call_seen = True
            acc.feed(delta)

            # Ordinary ASR+LLM turns follow the native tool protocol: visible
            # content is speech, while tool_calls are side-effect proposals.
            # Tools-disabled visual/retry requests use filtered plain content.
            if delta.content and not requires_structured_answer:
                visible = answer_filter.feed(delta.content)
                if visible:
                    if not tools_enabled:
                        yield {"type": "text", "content": visible}
                    elif not tool_call_seen:
                        pending_tool_text.append(visible)
        if not requires_structured_answer:
            visible_tail = answer_filter.flush()
            if not tools_enabled and visible_tail:
                yield {"type": "text", "content": visible_tail}
            elif tools_enabled and not tool_call_seen:
                if visible_tail:
                    pending_tool_text.append(visible_tail)

        tool_calls = acc.flush()
        direct_answers = [
            tc for tc in tool_calls if tc["name"] == DIRECT_ANSWER_TOOL_NAME
        ]
        if tools_enabled or requires_structured_answer:
            response_text = ""
            expression, intensity = "neutral", "low"
            fallback_actions = []
            structured_dialog_obtained = bool(direct_answers)
            if direct_answers:
                arguments = direct_answers[-1]["arguments"]
                candidate = arguments.get("response")
                if isinstance(candidate, str):
                    response_text = candidate.strip()
                expression, intensity = normalize_expression(
                    arguments.get("expression"), arguments.get("intensity")
                )
            elif self.settings.get("provider"):
                try:
                    response_text, expression, intensity, fallback_actions = self._json_dialog_answer(
                        messages,
                        request_model,
                        request_kwargs["max_tokens"],
                        [
                            tool for tool in tools
                            if tool.get("function", {}).get("name") != DIRECT_ANSWER_TOOL_NAME
                        ] if tools_enabled else [],
                    )
                    structured_dialog_obtained = True
                    LOGGER.warning(
                        "Model %s ignored direct_answer; accepted validated JSON fallback",
                        request_model,
                    )
                except Exception as exc:
                    LOGGER.warning("JSON dialog fallback failed: %s", exc)
            if fallback_actions and not any(
                call["name"] != DIRECT_ANSWER_TOOL_NAME for call in tool_calls
            ):
                tool_calls.extend(fallback_actions)
            if not response_text and tools_enabled and not tool_calls:
                response_text = "".join(pending_tool_text).strip()
            if not response_text and requires_structured_answer:
                raise StructuredAnswerUnavailableError(
                    "模型没有返回 direct_answer.response；请使用支持原生 Function Calling 的模型"
                )
            if response_text and tools_enabled and structured_dialog_obtained:
                yield {
                    "type": "dialog_expression",
                    "expression": expression,
                    "intensity": intensity,
                }
            if response_text:
                yield {"type": "text", "content": response_text}

        offered_action_names = {
            tool["function"]["name"]
            for tool in tools
            if tools_enabled and isinstance(tool.get("function"), dict)
        }
        for tc in tool_calls:
            if tc["name"] == DIRECT_ANSWER_TOOL_NAME:
                continue
            if not tools_enabled or tc["name"] not in offered_action_names:
                LOGGER.warning("Discarded unoffered tool call: %s", tc["name"])
                continue
            yield {
                "type": "tool_call", 
                "name": tc["name"], 
                "arguments": json.dumps(tc["arguments"])
            }

        yield {"type": "done", "finish_reason": finish_reason}
