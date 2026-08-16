# services/llm_service.py
import json
import yaml
from openai import OpenAI
from services.llm_output_filter import VisibleAnswerFilter
from services.llm_prompt import with_direct_speech_policy
from services.llm_request_options import reasoning_request_options
from services.tool_dispatcher import get_tools, ToolCallAccumulator


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
        request_kwargs = {
            "model": self.model,
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
            tools = get_tools()
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "auto"
        request_kwargs.update(reasoning_request_options(self.settings))
        response = self.client.chat.completions.create(**request_kwargs)

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
