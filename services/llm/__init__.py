"""LLM requests, tool dispatch, multimodal adapters, and voice-turn logic.

大模型层负责请求准备、流式响应、工具分发、多模态适配和语音轮次协作。它生成或
路由意图，但不直接驱动硬件；实际动作必须经过动作与编排层的契约。
"""
