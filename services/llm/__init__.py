"""LLM request policy, streaming results, tools, and multimodal adapters.

English
-------
This package prepares provider-compatible requests, filters visible answers,
accumulates streaming text/tool calls, manages conversation history, dispatches
declared tools, and adapts image/audio-capable models.  Model output is treated
as an untrusted proposal: action arguments still pass through intent guards,
the action contract, and orchestration before execution.  LLM code must not
publish motor/servo commands or embed provider details into hardware services.

中文
----
大模型层负责准备供应商兼容请求、过滤可见回答、累积流式文本和工具调用、管理对话历史、
分发已声明工具，并适配支持图像或音频的模型。模型输出只能视为“不可信提案”：动作参数
仍必须经过意图守卫、动作契约和编排层后才能执行。LLM 代码不得直接发布电机或舵机命令，
也不得把模型供应商细节写入硬件服务。
"""
