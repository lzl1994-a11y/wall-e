"""
[ZH] 大模型调用服务：封装 Kimi/OpenAI 接口，管理 API Key 和上下文历史。
     行为：接收 Prompt 和动态工具单，返回大模型的流式文本与工具 JSON。
[EN] LLM Calling Service: Encapsulates Kimi/OpenAI API, manages API keys and chat history.
     Behavior: Receives Prompts and dynamic tool lists, yields streaming text and tool JSON.
"""
# services/llm_service.py
import yaml
from openai import OpenAI

# [ZH] 直接从 MCP 服务获取动态生成的技能表
# [EN] Dynamically fetch generated skills from the MCP service
import services.mcp_service as mcp

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
            tools=mcp.get_chat_tools(), 
            
            tool_choice="auto",
            temperature=self.settings.get('temperature', 0.3),
            
            # [ZH] 核心：开启流式输出
            # [EN] Core: Enable streaming output
            stream=True 
        )

        # [ZH] 用于临时存储碎片化的工具调用数据
        # [EN] Buffer for temporarily storing fragmented tool call data
        tool_calls_buffer = {}

        for chunk in response:
            delta = chunk.choices[0].delta

            # ==========================================
            # [ZH] 动作通道 (处理工具指令)
            # [EN] Action Channel (Processing tool commands)
            # ==========================================
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    # [ZH] 获取碎片的索引 (应对多工具并发调用)
                    # [EN] Get chunk index (to handle concurrent multi-tool calls)
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"name": tc.function.name, "arguments": ""}
                    
                    # [ZH] 拼装 JSON 字符串碎片
                    # [EN] Assemble JSON string chunks
                    if tc.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc.function.arguments

            # ==========================================
            # [ZH] 语音通道 (处理正常聊天文本)
            # [EN] Voice Channel (Processing normal chat text)
            # ==========================================
            if delta.content:
                # [ZH] 直接 yield 出文本，清洗工作交给外层流水线
                # [EN] Yield text directly, leaving filtering to the outer pipeline
                yield {"type": "text", "content": delta.content}

        # ==========================================
        # [ZH] 扫尾工作 (分发组装好的动作)
        # [EN] Cleanup (Dispatch assembled actions)
        # ==========================================
        for idx, tool_data in tool_calls_buffer.items():
            yield {
                "type": "tool_call", 
                "name": tool_data["name"], 
                "arguments": tool_data["arguments"]
            }