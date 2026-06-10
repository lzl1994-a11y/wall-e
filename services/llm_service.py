# services/llm_service.py
import json
import yaml
from openai import OpenAI
from services.tool_dispatcher import get_tools, ToolCallAccumulator


class LLMService:
    def __init__(self, config_path="core/config.yaml"):
        # [ZH] 1. 从配置文件读取配置
        # [EN] 1. Load configuration from YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.settings = config['ai_settings']
        self.system_prompt = config['system_prompt']
        
        # [ZH] 2. 初始化 OpenAI 客户端 (在此项目中指向 Kimi 的接口)
        # [EN] 2. Initialize OpenAI client (pointing to Kimi API in this project)
        self.client = OpenAI(
            api_key=self.settings['api_key'],
            base_url=self.settings['base_url']
        )
        self.model = self.settings['model']

    def chat_stream(self, user_text, chat_history=None):
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
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_text})

        # [ZH] 发起长连接流式请求
        # [EN] Send long-connection streaming request
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            
            # [ZH] 挂载 FastMCP 自动生成的工具表
            # [EN] Mount auto-generated tools from FastMCP
            tools=get_tools(), 
            
            tool_choice="auto",
            temperature=self.settings.get('temperature', 0.3),
            
            stream=True 
        )

        acc = ToolCallAccumulator()

        for chunk in response:
            delta = chunk.choices[0].delta

            acc.feed(delta)

            if delta.content:
                yield {"type": "text", "content": delta.content}

        for tc in acc.flush():
            yield {
                "type": "tool_call", 
                "name": tc["name"], 
                "arguments": json.dumps(tc["arguments"])
            }