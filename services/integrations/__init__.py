"""Externally exposed Web and MCP entry points and trust boundaries.

English
-------
Integration services terminate protocols used outside the process: the
configuration Web UI and the restricted MCP gateway.  They validate input,
apply authentication/allow-list rules, translate requests into internal service
calls, and serialize responses.  They must not duplicate action policy or
device drivers, and must never expose arbitrary ROS topics, filesystem access,
or shell execution merely because an external protocol can carry free-form data.

中文
----
集成层终止进程外部使用的协议，目前包括配置 Web UI 和受限 MCP 网关。它负责校验输入、
执行鉴权与白名单规则、把请求转换为内部服务调用并序列化响应。这里不得复制动作策略或
设备驱动，也不能因为外部协议能够携带自由文本，就暴露任意 ROS Topic、文件系统访问或
Shell 执行能力；这些限制共同构成项目的外部信任边界。
"""
