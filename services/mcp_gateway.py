"""Network MCP gateway configuration and tool surface for Wali.

The internal voice pipeline continues to use OpenAI-compatible function calls.
This module exposes a separate, deliberately small MCP surface for authorized
external agents. Tool handlers delegate to an injected ROS executor so the MCP
transport can be tested without ROS or hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import yaml
from fastmcp import FastMCP
from pydantic import Field

from services.action_intent_guard import validate_action_arguments


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "core" / "config.yaml"
MCP_TOKEN_ENV = "WALI_MCP_TOKEN"


class RobotActionExecutor(Protocol):
    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class McpGatewaySettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5555
    path: str = "/mcp"
    command_timeout_sec: float = 12.0

    @property
    def is_loopback(self) -> bool:
        return self.host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def load_mcp_gateway_settings(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> McpGatewaySettings:
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    raw = config.get("mcp", {}) if isinstance(config, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    enabled = raw.get("enabled", False)
    host = raw.get("host", "127.0.0.1")
    port = raw.get("port", 5555)
    path = raw.get("path", "/mcp")
    timeout = raw.get("command_timeout_sec", 12.0)

    if not isinstance(enabled, bool):
        enabled = False
    if not isinstance(host, str) or not host.strip():
        host = "127.0.0.1"
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        port = 5555
    if not isinstance(path, str) or not path.startswith("/"):
        path = "/mcp"
    path = path.rstrip("/") or "/mcp"
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 1.0 <= float(timeout) <= 120.0
    ):
        timeout = 12.0

    return McpGatewaySettings(
        enabled=enabled,
        host=host.strip(),
        port=port,
        path=path,
        command_timeout_sec=float(timeout),
    )


def require_safe_transport(settings: McpGatewaySettings, token: str | None) -> None:
    """Fail closed when a network-visible control endpoint has no credential."""
    if token is not None:
        try:
            token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(f"{MCP_TOKEN_ENV} 必须使用 ASCII 字符") from exc
        if len(token) < 32 or len(token) > 512 or any(char.isspace() for char in token):
            raise RuntimeError(
                f"{MCP_TOKEN_ENV} 必须是 32-512 位且不含空白字符的随机令牌"
            )
    if not settings.is_loopback and token is None:
        raise RuntimeError(
            f"MCP监听 {settings.host} 时必须在 config.yaml 的 mcp.token 中设置令牌"
            f"（兼容方式：{MCP_TOKEN_ENV}）"
        )


def _auth_provider(token: str | None):
    if not token:
        return None
    # Keep auth imports out of the unprotected loopback/test path. FastMCP's
    # verifier validates the Authorization bearer token before tool dispatch.
    from fastmcp.server.auth import StaticTokenVerifier

    return StaticTokenVerifier(
        tokens={
            token: {
                "client_id": "wali-mcp-client",
                "scopes": ["wali:control"],
            }
        },
        required_scopes=["wali:control"],
    )


def create_mcp_gateway(
    executor: RobotActionExecutor,
    settings: McpGatewaySettings,
    *,
    token: str | None = None,
) -> FastMCP:
    """Create a stateless Streamable HTTP MCP server over a ROS executor."""
    require_safe_transport(settings, token)
    server = FastMCP(
        "Wali Robot Control",
        instructions=(
            "This server controls a physical robot. Use movement tools only for an "
            "explicit current user command. Never infer or repeat physical actions. "
            "The server enforces short durations and may reject commands."
        ),
        auth=_auth_provider(token),
        strict_input_validation=True,
        mask_error_details=True,
    )

    def invoke(name: str, arguments: dict[str, Any], *, extra_wait: float = 0.0):
        allowed, reason = validate_action_arguments(name, arguments)
        if not allowed:
            return {"status": "rejected", "action": name, "reason": reason}
        return executor.execute(
            name,
            arguments,
            timeout=settings.command_timeout_sec + max(0.0, extra_wait),
        )

    state_change = {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    physical_motion = {**state_change, "destructiveHint": True}

    @server.tool(annotations=physical_motion)
    def move_chassis(
        direction: Literal["forward", "backward", "spin", "left", "right"],
        duration: Annotated[int, Field(ge=1, le=3)] = 1,
    ) -> dict[str, Any]:
        """Move Wali's chassis for 1-3 seconds after an explicit user command."""
        return invoke(
            "move_chassis",
            {"direction": direction, "duration": duration},
            extra_wait=float(duration),
        )

    @server.tool(annotations=state_change)
    def play_sequence(sequence_name: str) -> dict[str, Any]:
        """Play one configured head, eye, or arm performance sequence."""
        return invoke("play_sequence", {"sequence_name": sequence_name})

    @server.tool(annotations=state_change)
    def express_emotion(
        emotion: Literal["curious", "happy", "sad", "surprised", "disdain", "angry"],
    ) -> dict[str, Any]:
        """Express one requested emotion using Wali's body and screen."""
        return invoke("express_emotion", {"emotion": emotion})

    @server.tool(annotations=state_change)
    def set_tracking_mode(
        mode: Literal["follow_me", "look_at_me", "idle"],
    ) -> dict[str, Any]:
        """Start or stop Wali's visual tracking mode."""
        return invoke("set_tracking_mode", {"mode": mode})

    @server.tool(annotations=state_change)
    def set_vision_gate(enabled: bool) -> dict[str, Any]:
        """Enable or disable Wali's visual tracking subsystem."""
        return invoke("set_vision_gate", {"enabled": enabled})

    @server.tool(annotations=state_change)
    def control_music(
        action: Literal["play", "stop"],
        track: Annotated[str, Field(max_length=200)] = "",
    ) -> dict[str, Any]:
        """Play or stop a track from Wali's local music directory."""
        return invoke("control_music", {"action": action, "track": track})

    @server.tool(annotations={
        **state_change,
        "idempotentHint": True,
    })
    def stop_all() -> dict[str, Any]:
        """Stop MCP-controlled chassis motion and interrupt the active sequence."""
        return invoke("stop_all", {})

    return server


def token_from_environment() -> str | None:
    """Load the configured MCP token; an environment value remains a fallback."""
    try:
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        mcp = config.get("mcp", {}) if isinstance(config, dict) else {}
        token = mcp.get("token") if isinstance(mcp, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    except (OSError, yaml.YAMLError):
        pass
    value = os.environ.get(MCP_TOKEN_ENV, "").strip()
    return value or None
