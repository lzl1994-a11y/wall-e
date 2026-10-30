"""Bounded, grounded plans joining existing actions and visual workflows.

The matcher only selects a planning path. It never translates words into
motions; the model proposes the complete ordered list and the runtime validates
every step before allowing the first side effect.
"""

from __future__ import annotations

import re
from typing import Any

from services.action.action_intent_guard import (
    explicit_sequence_mentions,
    is_non_execution_request,
    registered_sequence_names,
    validate_action_arguments,
    validate_action_call,
)
from services.dialog.dialog_expression_protocol import EXPRESSIONS


ORDERED_TASK_TOOL_NAME = "run_ordered_task"
MAX_ORDERED_TASK_STEPS = 8
ORDERED_TASK_PLANNING_POLICY = (
    "请理解本轮完整目标，使用可用技能规划完成目标所需的步骤，允许推导必要准备动作。"
    "翻译、复述、朗读、引用、解释、能力询问只回答，不调用执行工具。"
    "steps 必须保留用户要求的全部动作、观察问题、照片保存和原顺序，每步 grounding"
    "使用支持该步的连续短分句，按原话位置递增且不复用整句。‘然后再’只是顺序连接，不产生空步骤；"
    "‘告诉我结果’由最后回答完成，不另造步骤。无法完整规划时澄清，不能执行残缺任务。"
    "低头=play_sequence(head_down)，挥手=play_sequence(wave_hello)；"
    "‘低头、挥手、再看看并保存’应有低头、挥手、inspect_camera 三步，"
    "最后一项 question 保留观察问题且 save_photo=true，不能漏掉保存。"
    "‘转身再找红盒子，找到后挥手’应先 move_chassis(direction=spin,duration=1)，"
    "再 search_environment(target=红盒子,on_found_actions=[{name:play_sequence,"
    "arguments:{sequence_name:wave_hello}}])。搜索本身已负责必要转向，用户未要求"
    "先转身时不额外添加。当前转向按时长执行，不能承诺精确180度等角度；要求精确"
    "角度时说明限制并澄清。找到才做的动作放入 search_environment.on_found_actions，"
    "搜索后的外层步骤表示无论找到与否都执行的动作，不得把条件动作改成无条件。"
)
ORDERED_ACTION_TOOLS = frozenset({
    "control_music", "express_emotion", "move_chassis", "play_sequence",
    "play_choreography", "set_tracking_mode", "set_vision_gate", "stop_all",
})
ORDERED_STEP_TOOLS = ORDERED_ACTION_TOOLS | {"inspect_camera", "search_environment"}
_MOTION_CUE = re.compile(
    r"低头|把头低|抬头|转头|扭头|转身|转过身|转过去|转一圈|转个圈|转圈|"
    r"转(?:向|到|一下)|左转|右转|向左|向右|往左|往右|回正|挥手|招手|抬手|举手|"
    r"点头|摇头|放下.*手|前进|后退|往前|往后|向前|向后|走.*步|跳舞|"
    r"(?:抬|举)(?:起|一下)?(?:左|右|双|两只).*(?:手|胳膊)|(?:左|右|双|两只).*手.*(?:抬|举)"
)
_ORDER_CUE = re.compile(r"先|然后|再|接着|之后|以后|随后|并且|同时")
_VISUAL_CUE = re.compile(
    r"看看|看一看|看一下|看下|看一眼|才能看|才看|观察|识别|辨认|拍照|拍.*照片|"
    r"拍.*照|找|寻找|搜索|在哪|有什么|瞅|瞧"
)
_TURN_AROUND = re.compile(r"转身|转过身|转过去|掉头|转(?:一|个)?圈")
_FOUND_ONLY_PREFIX = re.compile(
    r"(?:找到(?:了)?(?:后|以后|就)|发现(?:目标|它|了)?(?:后|就))"
    r"[，,：:]*(?:请|你|再|然后|帮我|给我|先)*$"
)


def _grounding_spans(user_prompt: str, steps: list[dict[str, Any]]) -> list[tuple[int, int]]:
    compact = "".join(str(user_prompt or "").split())
    spans = []
    next_start = 0
    for index, step in enumerate(steps, start=1):
        grounding = "".join(step["grounding"].split())
        start = compact.find(grounding, next_start)
        if start < 0:
            raise ValueError(f"ordered_step_{index}_grounding_order_invalid")
        spans.append((start, start + len(grounding)))
        next_start = start + 1
    return spans


def _validate_complete_plan(user_prompt: str, plan: dict[str, Any]) -> None:
    """Reject known omissions without inventing any missing executable step."""
    from services.vision.camera_frame import is_camera_photo_request

    steps = plan["steps"]
    spans = _grounding_spans(user_prompt, steps)
    compact = "".join(str(user_prompt or "").split())
    actions = []
    seen_search = False
    for step, span in zip(steps, spans):
        actions.append((step["name"], step["arguments"], span))
        if step["name"] == "search_environment":
            seen_search = True
            actions.extend((child["name"], child["arguments"], span)
                           for child in step["arguments"]["on_found_actions"])
        elif seen_search and step["name"] in ORDERED_ACTION_TOOLS:
            prefix = compact[max(0, span[0] - 30):span[0]]
            if _FOUND_ONLY_PREFIX.search(prefix) or re.search(
                r"找到(?:了)?(?:后|以后|就)", step["grounding"],
            ):
                raise ValueError("ordered_found_only_action_requires_nested_plan")

    equivalent_sequences = {"wave_hello": {"wave_hello", "basic_wave"}}

    def observation_mention(start: int, end: int) -> bool:
        # A person lowering their head may be the subject of the camera
        # question, rather than another instruction for the robot to move.
        for step, span in zip(steps, spans):
            if (step["name"] not in {"inspect_camera", "search_environment"}
                    or not (span[0] <= start and end <= span[1])):
                continue
            prefix = compact[span[0]:start]
            if not _VISUAL_CUE.search(prefix):
                continue
            if step["name"] == "search_environment" and re.search(
                r"找到(?:了)?(?:后|以后|就)|发现(?:目标|它|了)?(?:后|就)", prefix,
            ):
                continue
            return True
        return False

    for expected, start, end in explicit_sequence_mentions(user_prompt):
        if observation_mention(start, end):
            continue
        choices = equivalent_sequences.get(expected, {expected})
        if not any(name == "play_sequence" and args.get("sequence_name") in choices
                   and span[0] <= start and end <= span[1]
                   for name, args, span in actions):
            raise ValueError(f"ordered_plan_missing_explicit_motion:{expected}")
    for match in _TURN_AROUND.finditer(compact):
        if observation_mention(match.start(), match.end()):
            continue
        if not any(name == "move_chassis" and args.get("direction") == "spin"
                   and span[0] <= match.start() and match.end() <= span[1]
                   for name, args, span in actions):
            raise ValueError("ordered_plan_missing_explicit_turn")
    photo_mentions = list(re.finditer(r"拍(?:张|一张|个)?照|拍一张|照张相|照一张|摄影", compact))
    observed_photo_only = bool(photo_mentions) and all(
        observation_mention(match.start(), match.end())
        and re.search(r"(?:有没有|是否|是不是).{0,8}$|(?:正在|在)$", compact[max(0, match.start() - 20):match.start()])
        for match in photo_mentions
    )
    explicit_save = bool(re.search(
        r"(?:保存|存下).{0,6}(?:照片|图片|相片)|(?:照片|相片).{0,6}(?:保存|存下)", compact,
    ))
    needs_photo_save = explicit_save or (is_camera_photo_request(user_prompt) and not observed_photo_only)
    if needs_photo_save and not any(
        step["name"] == "inspect_camera" and step["arguments"].get("save_photo") is True
        for step in steps
    ):
        raise ValueError("ordered_plan_missing_photo_save")
    if is_ordered_task_request(user_prompt) and not any(
        step["name"] in {"inspect_camera", "search_environment"} for step in steps
    ):
        raise ValueError("ordered_plan_missing_visual_step")


def is_ordered_task_request(text: Any) -> bool:
    """Select semantic planning for motion-plus-vision without choosing a motion."""
    compact = "".join(str(text or "").split())
    return bool(compact and not is_non_execution_request(compact)
                and _VISUAL_CUE.search(compact)
                and (_MOTION_CUE.search(compact) or _ORDER_CUE.search(compact)))


def normalize_ordered_task_plan(value: Any, *, user_prompt: str | None = None) -> dict[str, Any]:
    """Validate the complete shape, arguments and optional literal grounding."""
    if not isinstance(value, dict) or set(value) != {"steps"}:
        raise ValueError("ordered_plan_fields_invalid")
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_ORDERED_TASK_STEPS:
        raise ValueError("ordered_plan_steps_invalid")
    compact_prompt = "".join(str(user_prompt or "").split())
    normalized = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or set(step) != {"name", "arguments", "grounding"}:
            raise ValueError(f"ordered_step_{index}_fields_invalid")
        name, arguments, grounding = (step.get(key) for key in ("name", "arguments", "grounding"))
        if not isinstance(name, str) or name not in ORDERED_STEP_TOOLS:
            raise ValueError(f"ordered_step_{index}_unknown_tool")
        if not isinstance(grounding, str) or not grounding.strip() or len(grounding) > 500:
            raise ValueError(f"ordered_step_{index}_grounding_invalid")
        grounding = grounding.strip()
        if user_prompt is not None and "".join(grounding.split()) not in compact_prompt:
            raise ValueError(f"ordered_step_{index}_ungrounded")
        allowed, reason = validate_action_arguments(name, arguments)
        if not allowed:
            raise ValueError(f"ordered_step_{index}_{reason}")
        arguments = dict(arguments)
        if name == "search_environment":
            from services.vision.visual_search import normalize_visual_search_arguments
            arguments = normalize_visual_search_arguments(arguments)
            for child in arguments["on_found_actions"]:
                if child["name"] not in ORDERED_ACTION_TOOLS:
                    raise ValueError(f"ordered_step_{index}_completion_action_not_allowed")
                allowed, reason = validate_action_arguments(child["name"], child["arguments"])
                if not allowed:
                    raise ValueError(f"ordered_step_{index}_completion_{reason}")
        normalized.append({"name": name, "arguments": arguments, "grounding": grounding})
    return {"steps": normalized}


def validate_ordered_task_plan(user_prompt: str, value: Any) -> dict[str, Any]:
    """Authorize every grounded step atomically before execution can start."""
    if is_non_execution_request(user_prompt):
        raise ValueError("ordered_plan_non_command_context")
    plan = normalize_ordered_task_plan(value, user_prompt=user_prompt)
    _validate_complete_plan(user_prompt, plan)
    for index, step in enumerate(plan["steps"], start=1):
        # The full request preserves negation/quotation scope. The literal
        # clause then checks the direction or sequence selected for this step.
        allowed, reason = validate_action_call(user_prompt, step["name"], step["arguments"])
        # Distinct ordered clauses can contain different directions/durations.
        # Global context rejects negation; concrete arguments belong to the
        # literal clause rather than the first duration found in the full text.
        if not allowed and reason != "argument_conflict":
            raise ValueError(f"ordered_step_{index}_{reason}")
        allowed, reason = validate_action_call(step["grounding"], step["name"], step["arguments"])
        if not allowed:
            raise ValueError(f"ordered_step_{index}_{reason}")
        if step["name"] == "search_environment":
            for child in step["arguments"]["on_found_actions"]:
                allowed, reason = validate_action_call(step["grounding"], child["name"], child["arguments"])
                if not allowed:
                    raise ValueError(f"ordered_step_{index}_completion_{reason}")
    return plan


def ordered_task_tool_schema() -> dict[str, Any]:
    """Return a model-visible schema for complete mixed task plans."""
    argument_properties = {
        "sequence_name": {"type": "string", "enum": list(registered_sequence_names())},
        "choreography_id": {"type": "string", "maxLength": 64},
        "emotion": {"type": "string", "enum": sorted(EXPRESSIONS)},
        "mode": {"type": "string", "enum": ["follow_me", "look_at_me", "idle"]},
        "enabled": {"type": "boolean"},
        "direction": {"type": "string", "enum": ["forward", "backward", "spin", "left", "right"]},
        "duration": {"type": "integer", "minimum": 1, "maximum": 3},
        "action": {"type": "string", "enum": ["play", "stop"]},
        "track": {"type": "string", "maxLength": 200},
        "question": {"type": "string", "maxLength": 500},
        "save_photo": {"type": "boolean", "description": "inspect_camera: true 表示保存照片；question 非空时仍需回答画面问题"},
        "target": {"type": "string", "maxLength": 200},
        "search_direction": {"type": "string", "enum": ["spin", "left", "right"]},
        "motion_duration": {"type": "integer", "minimum": 1, "maximum": 3},
        "max_views": {"type": "integer", "minimum": 2, "maximum": 4},
    }
    argument_properties["on_found_actions"] = {
        "type": "array", "maxItems": 7,
        "description": "search_environment: 仅找到目标后才执行的动作，按原顺序保留",
        "items": {"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(ORDERED_ACTION_TOOLS)},
            "arguments": {"type": "object", "properties": {
                key: argument_properties[key] for key in (
                    "sequence_name", "choreography_id", "emotion", "mode", "enabled",
                    "direction", "duration", "action", "track",
                )
            }, "additionalProperties": False},
        }, "required": ["name", "arguments"], "additionalProperties": False},
    }
    return {
        "type": "object",
        "properties": {"steps": {
            "type": "array", "minItems": 1, "maxItems": MAX_ORDERED_TASK_STEPS,
            "description": "用户完整请求的全部步骤，严格按原顺序，不得遗漏动作、观察或照片保存；最后报告结果不单列动作",
            "items": {"type": "object", "properties": {
                "name": {"type": "string", "enum": sorted(ORDERED_STEP_TOOLS)},
                "arguments": {"type": "object", "properties": argument_properties, "additionalProperties": False,
                              "description": "只填对应工具参数；play_sequence 用 sequence_name，move_chassis 用 direction/duration，inspect_camera 用 question/save_photo，search_environment 用 target/question/search_direction/motion_duration/max_views"},
                "grounding": {"type": "string", "minLength": 1, "maxLength": 500,
                              "description": "原话中支持此步骤的连续短分句，按原话起点位置递增，不复用整句；不得改写或凭空补充"},
            }, "required": ["name", "arguments", "grounding"], "additionalProperties": False},
        }},
        "required": ["steps"], "additionalProperties": False,
    }


__all__ = ["ORDERED_TASK_TOOL_NAME", "ORDERED_TASK_PLANNING_POLICY", "ORDERED_ACTION_TOOLS", "ORDERED_STEP_TOOLS",
           "MAX_ORDERED_TASK_STEPS", "is_ordered_task_request",
           "normalize_ordered_task_plan", "validate_ordered_task_plan", "ordered_task_tool_schema"]
