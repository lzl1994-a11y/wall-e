"""Bounded, grounded plans joining existing actions and visual workflows.

The matcher only selects a planning path. It never translates words into
motions; the model proposes the complete ordered list and the runtime validates
every step before allowing the first side effect.
"""

from __future__ import annotations

import re
from typing import Any

from services.action.action_intent_guard import (
    explicit_action_mentions,
    explicit_sequence_mentions,
    is_non_execution_request,
    registered_sequence_names,
    validate_action_arguments,
    validate_action_call,
)
from services.dialog.dialog_expression_protocol import EXPRESSIONS
from services.orchestration.task_text import (
    clause_context, compact_text, has_negation, task_clause_spans,
)


ORDERED_TASK_TOOL_NAME = "run_ordered_task"
MAX_ORDERED_TASK_STEPS = 8
ORDERED_TASK_PLANNING_POLICY = (
    "请理解本轮完整请求。只有用户明确要求现在执行时才提交 run_ordered_task，"
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
    "‘如果找到就挥手’与‘找到后挥手’使用相同找到分支，并保留前面的所有独立步骤。"
    "被禁止的动作不加入执行步骤；‘不要拍照’约束对应观察的 save_photo=false，"
    "不能因此遗漏仍被要求的低头、观察等步骤。grounding 只引用支持该步实际执行的原话，"
    "不要用禁止项充当执行步骤的依据。‘先低头看看不要拍照’的观察 grounding 是‘看看’，"
    "save_photo=false；‘不要拍照’不是观察步骤的 grounding，禁止项仍由完整原话校验。"
    "当前混合计划只支持搜索找到目标后的动作分支；其他附加条件无法完整表达时澄清，"
    "不能把它简化成仅找到目标，也不能先执行部分步骤。"
)
ORDERED_ACTION_TOOLS = frozenset({
    "control_music", "express_emotion", "move_chassis", "play_sequence",
    "play_choreography", "set_tracking_mode", "set_vision_gate", "stop_all",
})
ORDERED_STEP_TOOLS = ORDERED_ACTION_TOOLS | {"inspect_camera", "search_environment"}
_VISUAL_CUE = re.compile(
    r"看看|看一看|看一下|看下|看一眼|才能看|才看|观察|识别|辨认|拍照|拍.*照片|"
    r"拍.*照|找|寻找|搜索|在哪|有什么|瞅|瞧"
)
_TURN_AROUND = re.compile(r"转身|转过身|转过去|掉头|转(?:一|个)?圈")
_CONDITION_CUE = re.compile(
    r"如果|假如|要是|一旦|只要|当(?=.{1,80}时)|的话|"
    r"(?:看到|看见|检测到|识别到).{0,80}?(?:就|便|则)|"
    r"找到(?:了)?(?:后|以后|就|才)|"
    r"发现(?:目标|它|了)?(?:后|就|才)"
)
_PHOTO_OPERATION = r"拍(?:张|一张|个)?照|拍摄|照相|摄影|保存|存下"
_PHOTO_PROHIBITION = re.compile(
    rf"(?:别|不要|不许|不准|禁止|不用|不必|无需|不需要).{{0,6}}(?:{_PHOTO_OPERATION})|"
    rf"(?:{_PHOTO_OPERATION}).{{0,8}}(?:不用了?|不要了?|不必|算了|取消|作罢)"
)
_PURE_PHOTO_PROHIBITION = re.compile(
    rf"^(?:请|你|也|先|这次|现在|千万)*(?:{_PHOTO_PROHIBITION.pattern})(?:了|吧|啊|呀|哦)*$"
)


def _found_consequence(text: str, target: str) -> tuple[int, str] | None:
    """Recognize only the found-positive branch the existing search supports."""
    found = re.search(
        rf"(?:找到|发现)(?:了)?(?:{re.escape(target)}|目标|它)?"
        r"(?:以后|之后|后|(?:的话)?[，,]*(?:你)?(?:就|才))", text,
    )
    if found is None or re.search(r"(?:没|未|不|无法|没有)(?:能)?$", text[:found.start()]):
        return None
    # Additional predicates need a richer conditional plan, not a found-only
    # search branch. Do not silently weaken them to target visibility.
    prefix = text[:found.start()]
    condition = re.search(r"如果|假如|要是|一旦|只要", prefix)
    if condition is not None and prefix[condition.end():].strip("，,"):
        return None
    consequence = text[found.end():]
    # Only direct actions belong in this branch. An extra state predicate
    # (e.g. '且它没有破损再挥手') must not become merely 'found -> wave'.
    if re.search(r"且|而且|如果|假如|要是|一旦|只要|当.{1,80}时|的话|没有|是否", consequence):
        return None
    mentions = explicit_action_mentions(consequence)
    action_start = min((start for _, _, start, _ in mentions), default=None)
    if action_start is None:
        action = re.search(r"停止|关闭|打开|开启|播放|启用|退出|跟随", consequence)
        action_start = action.start() if action else None
    if action_start is None or not re.fullmatch(
        r"(?:请|你|再|然后|帮我|给我|先|接着|随后|最后|向我|朝我|对我|做个|做一个|做|[，,：:])*",
        consequence[:action_start],
    ):
        return None
    return found.end(), consequence


def _scope_for_step(user_prompt: str, span: tuple[int, int], name: str,
                    arguments: dict[str, Any]) -> str:
    """Recover original evidence; harmless photo constraints stay separate."""
    context = clause_context(user_prompt, *span)
    clauses = task_clause_spans(context)
    if name == "inspect_camera":
        # A model may quote the entire request, including an unrelated negative
        # movement clause. Keep the real observation clause(s), not that whole
        # quote. A denied observation is retained and rejected by authorization.
        relevant = [(left, right) for left, right in clauses
                    if (_VISUAL_CUE.search(context[left:right])
                        or re.fullmatch(r"(?:直接|先|再)?看(?:吧|一下)?", context[left:right])
                        or re.search(r"保存|存下", context[left:right]))
                    and not (not arguments.get("save_photo")
                             and _PURE_PHOTO_PROHIBITION.fullmatch(context[left:right]))]
        if not relevant:
            return ""
        context = context[relevant[0][0]:relevant[-1][1]]
        visual = _VISUAL_CUE.search(context)
        if visual and explicit_action_mentions(context[:visual.start()]) and not has_negation(context[:visual.start()]):
            context = context[visual.start():]
    elif name in ORDERED_ACTION_TOOLS:
        mentions = [(start, end) for tool, partial, start, end in explicit_action_mentions(context)
                    if tool == name and all(arguments.get(key) == value for key, value in partial.items())]
        if not mentions and name == "move_chassis" and arguments.get("direction") == "spin":
            mentions = [(match.start(), match.end()) for match in _TURN_AROUND.finditer(context)]
        if mentions:
            # One physical step refers to one original instruction. Its
            # surrounding negation stays attached even if grounding omitted it.
            context = clause_context(context, *min(mentions))
    return context


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
    branches = []
    for step, span in zip(steps, spans):
        actions.append((step["name"], step["arguments"], span))
        if step["name"] == "search_environment":
            found = _found_consequence(compact[span[0]:span[1]], step["arguments"]["target"])
            if step["arguments"]["on_found_actions"] and found is None:
                raise ValueError("ordered_search_completion_requires_found_condition")
            if found is not None:
                branches.append((span[0], span[0] + found[0], span[1]))
            actions.extend((child["name"], child["arguments"], span)
                           for child in step["arguments"]["on_found_actions"])
        elif branches:
            # A found-only instruction cannot be moved outside its search.
            # Explicit 'regardless' wording is required to leave that branch.
            _, consequence, _ = branches[-1]
            if span[0] >= consequence and not re.search(
                r"无论|不管|不论", compact[consequence:span[1]],
            ):
                raise ValueError("ordered_found_only_action_requires_nested_plan")
    for condition in _CONDITION_CUE.finditer(compact):
        if not any(left <= condition.start() < consequence <= right
                   for left, consequence, right in branches):
            if any(step["name"] == "search_environment" and span[0] < condition.start()
                   for step, span in zip(steps, spans)):
                raise ValueError("ordered_found_only_action_requires_nested_plan")
            raise ValueError("ordered_condition_not_supported")

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
            if step["name"] == "search_environment" and any(
                left == span[0] and start >= consequence for left, consequence, _ in branches
            ):
                continue
            return True
        return False

    for expected, start, end in explicit_sequence_mentions(user_prompt):
        if observation_mention(start, end):
            continue
        choices = equivalent_sequences.get(expected, {expected})
        if has_negation(clause_context(compact, start, end)):
            continue
        if not any(name == "play_sequence" and args.get("sequence_name") in choices
                   and span[0] <= start and end <= span[1]
                   for name, args, span in actions):
            raise ValueError(f"ordered_plan_missing_explicit_motion:{expected}")
    for tool, partial_args, start, end in explicit_action_mentions(user_prompt):
        if observation_mention(start, end):
            continue
        context = clause_context(compact, start, end)
        if not has_negation(context):
            continue
        future_only = bool(re.search(r"(?:不要|别|不用|不必)再", context))
        for name, args, span in actions:
            if future_only and span[1] <= start:
                continue
            matches = all(args.get(key) == value or (
                key == "sequence_name" and value == "wave_hello" and args.get(key) == "basic_wave"
            ) for key, value in partial_args.items())
            if name == tool and matches:
                raise ValueError("ordered_plan_negated_action")
    for match in _TURN_AROUND.finditer(compact):
        if observation_mention(match.start(), match.end()):
            continue
        if has_negation(clause_context(compact, match.start(), match.end())):
            if any(name == "move_chassis" and args.get("direction") == "spin"
                   for name, args, _ in actions):
                raise ValueError("ordered_plan_negated_action")
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
    inspections = [(index, span) for index, (step, span) in enumerate(zip(steps, spans))
                   if step["name"] == "inspect_camera"]
    for left, right in task_clause_spans(compact):
        clause = compact[left:right]
        if _PHOTO_PROHIBITION.search(clause):
            # A trailing storage constraint belongs to the current observation;
            # a leading constraint covers upcoming observations. Do not cancel
            # a distinct earlier photo when the user later asks only to look.
            previous = [(index, span) for index, span in inspections if span[0] < right]
            affected = previous[-1:] if previous else inspections
            if re.search(r"(?:不要|别|不用|不必)再", clause):
                affected = [(index, span) for index, span in inspections if span[1] > left]
            if any(steps[index]["arguments"].get("save_photo") for index, _ in affected):
                raise ValueError("ordered_plan_photo_save_forbidden")
            continue
        if has_negation(clause):
            continue
        explicit_save = re.search(
            r"(?:保存|存下).{0,6}(?:照片|图片|相片)|(?:照片|相片).{0,6}(?:保存|存下)", clause,
        )
        if explicit_save or (is_camera_photo_request(clause) and not observed_photo_only):
            if not any(steps[index]["arguments"].get("save_photo") is True
                       and span[0] < right and left < span[1] for index, span in inspections):
                raise ValueError("ordered_plan_missing_photo_save")
    if is_ordered_task_request(user_prompt) and not any(
        step["name"] in {"inspect_camera", "search_environment"} for step in steps
    ):
        raise ValueError("ordered_plan_missing_visual_step")


def is_ordered_task_request(text: Any) -> bool:
    """Select semantic planning for motion-plus-vision without choosing a motion."""
    from services.orchestration.task_routing import select_task_route

    return select_task_route(text) == "ordered_task"


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
    spans = _grounding_spans(user_prompt, plan["steps"])
    contexts = []
    for step, span in zip(plan["steps"], spans):
        context = _scope_for_step(user_prompt, span, step["name"], step["arguments"])
        contexts.append(context)
        if (context and len(task_clause_spans(step["grounding"])) > 1
                and len(context) < len(compact_text(step["grounding"]))):
            # Carry the recovered original evidence through to BT's independent
            # action authorization, instead of reintroducing the broad quote.
            step["grounding"] = context
    _validate_complete_plan(user_prompt, plan)
    for index, (step, context) in enumerate(zip(plan["steps"], contexts), start=1):
        # Global execution context was checked once above. Recover the original
        # clause here so a model cannot remove '不要' from its literal evidence.
        if not context:
            reason = "observation_evidence_missing" if step["name"] == "inspect_camera" else "negated_action"
            raise ValueError(f"ordered_step_{index}_{reason}")
        if step["name"] == "search_environment":
            branch = _found_consequence(context, step["arguments"]["target"])
            search_context = context[:branch[0]] if branch else context
            search_context = re.sub(r"(?:如果|假如|要是|一旦|只要).*", "", search_context)
            allowed, reason = validate_action_call(search_context, step["name"], step["arguments"])
        else:
            allowed, reason = validate_action_call(context, step["name"], step["arguments"])
        if not allowed:
            raise ValueError(f"ordered_step_{index}_{reason}")
        if step["name"] == "search_environment":
            for child in step["arguments"]["on_found_actions"]:
                # The search owner keeps these actions conditional. Authorize
                # only the actual consequent, with its own negation retained.
                allowed, reason = validate_action_call(branch[1], child["name"], child["arguments"])
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
