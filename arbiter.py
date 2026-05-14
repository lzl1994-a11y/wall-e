"""
决策大脑：状态机中枢，调度事件。
典型链路：收到语音/文本 -> 调 LLM -> 调 MCP 工具或 TTS。
"""

from queue import Empty
from time import sleep


class Arbiter:
    def __init__(self, bus):
        self.bus = bus
        self.running = True
        self.state = "idle"

    def run(self) -> None:
        while self.running:
            self._handle_voice_events()
            self._handle_llm_responses()
            self._handle_mcp_results()
            self._handle_vision_events()
            self._handle_doa_events()
            sleep(0.01)

    def _handle_voice_events(self) -> None:
        try:
            event = self.bus.voice_events.get_nowait()
        except Empty:
            return

        prompt = event.get("text", "")
        if not prompt:
            return

        self.state = "thinking"
        self.bus.llm_requests.put(
            {
                "prompt": prompt,
                "tools": self.bus.skills,
                "context": event.get("context", {}),
            }
        )

    def _handle_llm_responses(self) -> None:
        try:
            response = self.bus.llm_responses.get_nowait()
        except Empty:
            return

        for tool_call in response.get("tool_calls", []):
            self.bus.mcp_tool_calls.put(tool_call)

        text = response.get("text")
        if text:
            self.bus.tts_requests.put({"text": text})

        if not response.get("tool_calls"):
            self.state = "idle"

    def _handle_mcp_results(self) -> None:
        try:
            result = self.bus.mcp_results.get_nowait()
        except Empty:
            return

        if result.get("speak"):
            self.bus.tts_requests.put({"text": result["speak"]})

        self.state = "idle"

    def _handle_vision_events(self) -> None:
        try:
            event = self.bus.vision_events.get_nowait()
        except Empty:
            return

        if event.get("type") == "target_error":
            self.bus.servo_targets.put(event["payload"])

    def _handle_doa_events(self) -> None:
        try:
            event = self.bus.doa_events.get_nowait()
        except Empty:
            return

        self.bus.motion_commands.put({"type": "turn_to_angle", "payload": event})
