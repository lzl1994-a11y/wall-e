"""
通讯总线：定义跨进程 Queue 和共享变量 Value。
所有服务互不直接通信，只通过 Arbiter 调度。
"""

from multiprocessing import Queue, Value


class MessageBus:
    def __init__(
        self,
        config,
        skills,
        voice_events,
        llm_requests,
        llm_responses,
        mcp_tool_calls,
        mcp_results,
        tts_requests,
        vision_events,
        doa_events,
        servo_targets,
        motion_commands,
        telemetry_events,
        vision_enabled,
    ):
        self.config = config
        self.skills = skills
        self.voice_events = voice_events
        self.llm_requests = llm_requests
        self.llm_responses = llm_responses
        self.mcp_tool_calls = mcp_tool_calls
        self.mcp_results = mcp_results
        self.tts_requests = tts_requests
        self.vision_events = vision_events
        self.doa_events = doa_events
        self.servo_targets = servo_targets
        self.motion_commands = motion_commands
        self.telemetry_events = telemetry_events
        self.vision_enabled = vision_enabled


def create_message_bus(config=None) -> MessageBus:
    config = config or {}
    return MessageBus(
        config=config,
        skills=config.get("wali_skills", []),
        voice_events=Queue(),
        llm_requests=Queue(),
        llm_responses=Queue(),
        mcp_tool_calls=Queue(),
        mcp_results=Queue(),
        tts_requests=Queue(),
        vision_events=Queue(),
        doa_events=Queue(),
        servo_targets=Queue(),
        motion_commands=Queue(),
        telemetry_events=Queue(),
        vision_enabled=Value("b", bool(config.get("vision", {}).get("enabled_on_start", False))),
    )
