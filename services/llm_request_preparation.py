"""Pure service for voice request routing and text LLM prompt preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from pypinyin import Style, pinyin

from services.action_intent_guard import deterministic_safety_action
from services.camera_frame import (
    is_camera_inspection_request,
    is_camera_photo_request,
)
from services.conditional_task import is_conditional_task_request
from services.llm_response_policy import LLMResponsePolicy

ROUTE_SAFETY_ACTION = "safety_action"
ROUTE_CONDITIONAL_TASK = "conditional_task"
ROUTE_CAMERA_PHOTO = "camera_photo"
ROUTE_CAMERA_INSPECTION = "camera_inspection"
ROUTE_ORDINARY_DIALOG = "ordinary_dialog"


@dataclass(frozen=True)
class PreparedVoiceRequest:
    route: str
    user_prompt: str
    safety_action_name: Optional[str] = None
    safety_action_arguments: Optional[dict[str, Any]] = None
    is_long_form: bool = False
    augmented_prompt: Optional[str] = None
    tools_enabled: bool = True
    max_tokens_override: Optional[int] = None

    @property
    def is_safety_action(self) -> bool:
        return self.route == ROUTE_SAFETY_ACTION

    @property
    def is_conditional_task(self) -> bool:
        return self.route == ROUTE_CONDITIONAL_TASK

    @property
    def is_camera_photo(self) -> bool:
        return self.route == ROUTE_CAMERA_PHOTO

    @property
    def is_camera_inspection(self) -> bool:
        return self.route == ROUTE_CAMERA_INSPECTION

    @property
    def is_ordinary_dialog(self) -> bool:
        return self.route == ROUTE_ORDINARY_DIALOG


class LLMRequestPreparation:
    LONG_FORM_POLICY = (
        "这是朗读、背诵或完整内容请求。请连续完整输出用户要求的正文，"
        "不要只给标题、简介或开头一句；除非用户明确只要片段。"
    )
    ORDINARY_DIALOG_POLICY = "普通对话保持一到两句、简短自然。"
    PROMPT_INSTRUCTION = (
        "请结合对话上下文和拼音静默理解用户本意，然后直接回答。"
        "只输出可以通过扬声器播放的最终台词，不要输出或复述原始 ASR 文本、"
        "拼音、修正文本、纠错结果、分析、思考、计划、规则复述、示例、列表、Markdown、"
        "括号说明、Function Calling 字样、工具名或工具参数。"
        "需要动作时只使用原生工具调用，不要在文字中描述调用过程。"
    )

    @classmethod
    def generate_pinyin_reference(cls, user_prompt: str) -> str:
        py_list = pinyin(user_prompt or "", style=Style.NORMAL)
        return " ".join([item[0] for item in py_list])

    @classmethod
    def build_augmented_prompt(
        cls, user_prompt: str, *, py_str: str, is_long_form: bool
    ) -> str:
        response_policy = (
            cls.LONG_FORM_POLICY if is_long_form else cls.ORDINARY_DIALOG_POLICY
        )
        return (
            f"原始 ASR 文本：{user_prompt}\n"
            f"拼音参考：{py_str}\n\n"
            f"{cls.PROMPT_INSTRUCTION}"
            f"{response_policy}"
        )

    @classmethod
    def resolve_configured_max_tokens(cls, configured_tokens: object) -> object:
        if isinstance(configured_tokens, Mapping):
            return configured_tokens.get("max_tokens", 0)
        return configured_tokens

    @classmethod
    def prepare_voice_request(
        cls,
        user_prompt: str,
        configured_tokens: object = 0,
        *,
        model_settings: Optional[Mapping[str, Any]] = None,
    ) -> PreparedVoiceRequest:
        text = str(user_prompt or "")

        # 1. Deterministic safety action
        safety_action = deterministic_safety_action(text)
        if safety_action is not None:
            action_name, action_arguments = safety_action
            return PreparedVoiceRequest(
                route=ROUTE_SAFETY_ACTION,
                user_prompt=user_prompt,
                safety_action_name=action_name,
                safety_action_arguments=action_arguments,
                is_long_form=False,
                augmented_prompt=None,
                tools_enabled=True,
                max_tokens_override=None,
            )

        # 2. Conditional task
        conditional_request = is_conditional_task_request(text)
        if conditional_request:
            return PreparedVoiceRequest(
                route=ROUTE_CONDITIONAL_TASK,
                user_prompt=user_prompt,
                safety_action_name=None,
                safety_action_arguments=None,
                is_long_form=False,
                augmented_prompt=None,
                tools_enabled=True,
                max_tokens_override=None,
            )

        # 3. Camera photo (not conditional)
        if is_camera_photo_request(text) and not conditional_request:
            return PreparedVoiceRequest(
                route=ROUTE_CAMERA_PHOTO,
                user_prompt=user_prompt,
                safety_action_name=None,
                safety_action_arguments=None,
                is_long_form=False,
                augmented_prompt=None,
                tools_enabled=True,
                max_tokens_override=None,
            )

        # 4. Camera inspection (not conditional)
        if is_camera_inspection_request(text) and not conditional_request:
            return PreparedVoiceRequest(
                route=ROUTE_CAMERA_INSPECTION,
                user_prompt=user_prompt,
                safety_action_name=None,
                safety_action_arguments=None,
                is_long_form=False,
                augmented_prompt=None,
                tools_enabled=True,
                max_tokens_override=None,
            )

        # 5. Ordinary dialog
        py_str = cls.generate_pinyin_reference(text)
        is_long_form = LLMResponsePolicy.is_long_form_request(text)
        augmented_prompt = cls.build_augmented_prompt(
            user_prompt, py_str=py_str, is_long_form=is_long_form
        )

        tokens_source = (
            model_settings if model_settings is not None else configured_tokens
        )
        raw_tokens = cls.resolve_configured_max_tokens(tokens_source)
        max_tokens_override = LLMResponsePolicy.max_tokens(
            raw_tokens, long_form=is_long_form
        )

        return PreparedVoiceRequest(
            route=ROUTE_ORDINARY_DIALOG,
            user_prompt=user_prompt,
            safety_action_name=None,
            safety_action_arguments=None,
            is_long_form=is_long_form,
            augmented_prompt=augmented_prompt,
            tools_enabled=True,
            max_tokens_override=max_tokens_override,
        )


def prepare_voice_request(
    user_prompt: str,
    configured_tokens: object = 0,
    *,
    model_settings: Optional[Mapping[str, Any]] = None,
) -> PreparedVoiceRequest:
    return LLMRequestPreparation.prepare_voice_request(
        user_prompt,
        configured_tokens=configured_tokens,
        model_settings=model_settings,
    )
