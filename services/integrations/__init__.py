"""Externally exposed web and MCP integration boundaries.

集成层承接 Web 与 MCP 等外部入口，负责协议适配、鉴权和请求转换。它调用内部服务，
但不重新实现业务规则或硬件驱动，从而把公网/局域网边界与核心逻辑隔离。
"""
