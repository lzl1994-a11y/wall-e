"""Pure service for visual model request preparation and response accumulation."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from services.llm_response_policy import LLMResponsePolicy

GAME_VISION_PROMPT = (
    "观察这张正在运行的 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
    "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。只输出可直接播报的一句话。"
)
GAME_VISION_SYSTEM_PROMPT = (
    "你是陪主人玩 FC 游戏的瓦力。只依据当前游戏截图简短评论，不输出分析过程。"
)
GAME_VISION_MAX_TOKENS = 96

CAMERA_QA_SYSTEM_PROMPT = (
    "你是瓦力的视觉。只依据当前摄像头图片回答问题；看不清时明确说看不清，"
    "不要猜测。答案使用简短自然的中文，不能输出分析过程或任何标签。"
)

CONDITIONAL_VISION_SYSTEM_PROMPT = (
    "你是机器人视觉条件判断器。只能依据当前图片返回严格 JSON；"
    "无法确认时必须返回 uncertain，禁止猜测。"
)
CONDITIONAL_VISION_MAX_TOKENS = 160


@dataclass(frozen=True)
class VisualModelRequest:
    prompt: str
    image_base64: str
    history: list[dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = False
    structured_answer: bool = False
    system_prompt: Optional[str] = None
    max_tokens_override: Optional[int] = None


def encode_image_base64(frame: bytes | bytearray) -> str:
    return base64.b64encode(frame).decode("ascii")


def build_game_vision_request(jpeg: bytes | bytearray) -> VisualModelRequest:
    return VisualModelRequest(
        prompt=GAME_VISION_PROMPT,
        history=[],
        image_base64=encode_image_base64(jpeg),
        tools_enabled=False,
        structured_answer=False,
        system_prompt=GAME_VISION_SYSTEM_PROMPT,
        max_tokens_override=GAME_VISION_MAX_TOKENS,
    )


def build_camera_qa_prompt(user_prompt: str) -> str:
    return (
        "请根据我附带的摄像头画面回答用户的问题。只输出简短、自然、可直接播报的中文答案，"
        "不要输出修正文本标签、分析过程、工具调用或括号说明。\n"
        f"用户问题：{user_prompt}"
    )


def build_camera_qa_request(
    frame: bytes | bytearray,
    *,
    user_prompt: str,
    history: Optional[Sequence[dict[str, Any]]] = None,
) -> VisualModelRequest:
    return VisualModelRequest(
        prompt=build_camera_qa_prompt(user_prompt),
        history=list(history) if history is not None else [],
        image_base64=encode_image_base64(frame),
        tools_enabled=False,
        structured_answer=False,
        system_prompt=CAMERA_QA_SYSTEM_PROMPT,
        max_tokens_override=None,
    )


def build_conditional_vision_prompt(observation: str, condition: str) -> str:
    return (
        "请只依据附带的当前摄像头画面判断条件。返回一个 JSON 对象，且只能包含 "
        "decision 和 evidence。decision 只能是 yes、no、uncertain；图片不足以确认时"
        "必须使用 uncertain。不要执行动作，不要输出 Markdown 或其他文字。\n"
        f"观察任务：{observation}\n判断条件：{condition}"
    )


def build_conditional_vision_request(
    frame: bytes | bytearray,
    *,
    observation: str,
    condition: str,
) -> VisualModelRequest:
    return VisualModelRequest(
        prompt=build_conditional_vision_prompt(observation, condition),
        history=[],
        image_base64=encode_image_base64(frame),
        tools_enabled=False,
        structured_answer=False,
        system_prompt=CONDITIONAL_VISION_SYSTEM_PROMPT,
        max_tokens_override=CONDITIONAL_VISION_MAX_TOKENS,
    )


class VisualResponseAccumulator:
    """Accumulate stream text events and format raw or cleaned visual answers."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def process_event(self, event: Any) -> None:
        if not isinstance(event, Mapping):
            return
        if event.get("type") == "text":
            content = event.get("content")
            if isinstance(content, str) and content:
                self._chunks.append(content)

    def raw_text(self) -> str:
        return "".join(self._chunks).strip()

    def clean_answer(self) -> str:
        return LLMResponsePolicy.clean_visual_answer(self.raw_text())


class LLMVisualRequest:
    GAME_VISION_PROMPT = GAME_VISION_PROMPT
    GAME_VISION_SYSTEM_PROMPT = GAME_VISION_SYSTEM_PROMPT
    GAME_VISION_MAX_TOKENS = GAME_VISION_MAX_TOKENS
    CAMERA_QA_SYSTEM_PROMPT = CAMERA_QA_SYSTEM_PROMPT
    CONDITIONAL_VISION_SYSTEM_PROMPT = CONDITIONAL_VISION_SYSTEM_PROMPT
    CONDITIONAL_VISION_MAX_TOKENS = CONDITIONAL_VISION_MAX_TOKENS

    @staticmethod
    def encode_image_base64(frame: bytes | bytearray) -> str:
        return encode_image_base64(frame)

    @classmethod
    def build_game_vision_request(
        cls, jpeg: bytes | bytearray
    ) -> VisualModelRequest:
        return build_game_vision_request(jpeg)

    @classmethod
    def build_camera_qa_request(
        cls,
        frame: bytes | bytearray,
        *,
        user_prompt: str,
        history: Optional[Sequence[dict[str, Any]]] = None,
    ) -> VisualModelRequest:
        return build_camera_qa_request(frame, user_prompt=user_prompt, history=history)

    @classmethod
    def build_conditional_vision_request(
        cls,
        frame: bytes | bytearray,
        *,
        observation: str,
        condition: str,
    ) -> VisualModelRequest:
        return build_conditional_vision_request(
            frame, observation=observation, condition=condition
        )
