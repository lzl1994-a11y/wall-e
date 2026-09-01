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
            if direct_answers:
                arguments = direct_answers[-1]["arguments"]
                candidate = arguments.get("response")
                if isinstance(candidate, str):
                    response_text = candidate.strip()
                expression, intensity = normalize_expression(
                    arguments.get("expression"), arguments.get("intensity")
                )
            if not response_text and tools_enabled and not tool_calls:
                response_text = "".join(pending_tool_text).strip()
            if not response_text and requires_structured_answer:
                raise StructuredAnswerUnavailableError(
                    "模型没有返回 direct_answer.response；请使用支持原生 Function Calling 的模型"
                )
            if response_text and tools_enabled and direct_answers:
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
