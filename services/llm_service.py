# services/llm_service.py
import json
import logging
import yaml
from openai import OpenAI
from services.llm_output_filter import VisibleAnswerFilter
from services.llm_prompt import with_direct_speech_policy
from services.llm_request_options import reasoning_request_options
from services.tool_dispatcher import get_tools, ToolCallAccumulator


LOGGER = logging.getLogger(__name__)
DEFAULT_TOOL_MODEL_BY_PRIMARY = {
    # These 4.1V thinking models may accept `tools` but return normal text
    # instead of a tool call. Use the tested tools-capable V model unless the
    # operator explicitly sets llm.tool_model.
    "glm-4.1v-thinking-flashx": "glm-4.6v-flash",
    "glm-4.1v-thinking-flash": "glm-4.6v-flash",
}


class ToolCallingUnavailableError(RuntimeError):
    """The configured model or MCP registry cannot service a tools-enabled turn."""


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
        fallback = DEFAULT_TOOL_MODEL_BY_PRIMARY.get(str(self.model).strip().lower())
        if fallback:
            return fallback, "known-model compatibility fallback"
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
        messages = [{
            "role": "system",
            "content": with_direct_speech_policy(selected_system_prompt),
        }]
        messages.extend(chat_history)
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
        request_model = self._request_model(tools_enabled)
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
        if tools_enabled:
            if self._tools is None:
                self._tools = get_tools()
            tools = self._tools
            if not tools:
                raise ToolCallingUnavailableError(
                    "动作工具为空；拒绝以无工具模式发送请求。请检查 FastMCP 2.x 工具注册。"
                )
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        request_settings = (
            self.settings
            if request_model == self.model
            else {**self.settings, "model": request_model}
        )
        request_kwargs.update(reasoning_request_options(request_settings))
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if tools_enabled and _is_tool_calling_rejection(exc):
                raise ToolCallingUnavailableError(
                    f"模型 {request_model!r} 拒绝或不支持 function calling；"
                    "请检查模型能力和兼容 API 的 tools 支持。"
                ) from exc
            raise

        acc = ToolCallAccumulator()
        answer_filter = VisibleAnswerFilter()
        finish_reason = ""

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = str(chunk_finish_reason)
            delta = choice.delta

            acc.feed(delta)

            if delta.content:
                visible = answer_filter.feed(delta.content)
                if visible:
                    yield {"type": "text", "content": visible}

        visible_tail = answer_filter.flush()
        if visible_tail:
            yield {"type": "text", "content": visible_tail}

        for tc in acc.flush():
            yield {
                "type": "tool_call", 
                "name": tc["name"], 
                "arguments": json.dumps(tc["arguments"])
            }

        yield {"type": "done", "finish_reason": finish_reason}
