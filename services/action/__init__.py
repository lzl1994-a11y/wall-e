"""Stable action contracts shared by every command producer and executor.

English
-------
This package defines the normalized action command, acknowledgement, status,
cancellation, arbitration, capability registry, and intent-safety rules.  It is
the boundary between callers that *request* behavior (dialog, MCP, gamepad, or
behavior trees) and components that execute it.  Code here may validate and
route an action, but it must not own a ROS node, publish directly to hardware,
or encode a PCA9685/serial protocol.  Keeping those concerns out makes the same
action contract usable in unit tests and with future transports.

中文
----
本包定义所有动作发起方和执行方共享的稳定契约，包括规范化动作命令、回执、状态、
定向取消、资源仲裁、能力注册表和意图安全校验。它位于“请求动作”的对话、MCP、手柄、
行为树与实际执行组件之间。这里可以校验和路由动作，但不得持有 ROS 节点生命周期、
直接向硬件发布数据，也不得编码 PCA9685 或串口协议。这样即使将来替换通信方式，
动作语义和安全规则仍然可以复用并独立测试。
"""
