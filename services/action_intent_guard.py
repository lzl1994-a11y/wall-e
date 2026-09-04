"""Safety validation for LLM-proposed robot actions.

The LLM is the positive semantic router.  This module deliberately does not
try to recognize every valid Chinese command a second time.  It validates the
tool shape and rejects only high-confidence unsafe contexts (negation,
quotation, third-party narration) or arguments that explicitly contradict the
user's words.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


_EMOTIONS = {"curious", "happy", "sad", "surprised", "disdain", "angry"}
_DIRECTIONS = {"forward", "backward", "spin", "left", "right"}
_TRACKING_MODES = {"follow_me", "look_at_me", "idle"}
_MUSIC_ACTIONS = {"play", "stop"}

_REQUEST_CUE_RE = re.compile(
    r"(?:请(?!问)|帮我|麻烦|给我|替我|为我|一下|好不好|好吗|[一二两三123]秒)"
)
_CAPABILITY_QUESTION_RE = re.compile(
    r"^(?:请问)?(?:你|瓦力)?(?:现在|目前|以后)?(?:会不会|会|能不能|能否|能|可不可以|可以|"
    r"有没有|有|是否|喜欢|知道|懂不懂).*(?:吗|么|能力|本事|功能|[?？])$"
)
_TRACKING_STOP_RE = re.compile(
    r"(?:停止|停下|退出|别再|不要再).*(?:跟着|跟随|看着|盯着|注视)|"
    r"(?:别|不要)(?:再)?(?:跟着|跟随|看着|盯着|注视|看了)|"
    r"(?:不想|不需要|不用).*(?:跟着|跟随|看着|盯着|注视)|"
    r"(?:跟着|跟随|看着|盯着|注视).*(?:不用|不要|算了|停止)"
)
_VISION_OFF_RE = re.compile(r"(?:关闭|关掉|停止|退出).*(?:视觉|跟踪)")
_LEADING_TRACKING_STOP_RE = re.compile(
    r"^(?:请|麻烦|帮我)?(?:停止|停下|退出|别(?:再)?|不要(?:再)?).{0,12}"
    r"(?:跟着|跟随|看着|盯着|注视|看)"
)
_LEADING_VISION_OFF_RE = re.compile(
    r"^(?:(?:请|麻烦|帮我)?(?:关闭|关掉|停止|退出).{0,12}(?:视觉|跟踪)|"
    r"(?:请|麻烦|帮我)?(?:把)?(?:视觉|跟踪).{0,8}(?:关闭|关掉|停止|退出))"
)
_CANCEL_STOP_RE = re.compile(
    r"(?:别|不要|不准|禁止)(?:再)?(?:停止|停下|退出|关闭|关掉).*"
    r"(?:跟着|跟随|看着|盯着|注视|视觉|跟踪)"
)
_CANCEL_EXPLICIT_STOP_RE = re.compile(
    r"(?:停止|停下|退出|关闭|关掉).*(?:不用了?|不要了?|不必|算了|取消|作罢)"
)
_NEGATED_ACTION_RE = re.compile(
    r"(?:别再?|不要再?|不准|禁止|停止|停下|不需要|不用|不想).*(?:"
    r"拍(?:张|一张|个)?照|拍摄|"
    r"观察|识别|打开|开启|关闭|关掉|走|挪|移动|前进|后退|左转|右转|转弯|"
    r"转圈|转头|歪头|低头|点头|抬手|举手|放下|挥手|招手|跳舞|表达|"
    r"做动作|做.*表情|表情|思考|倾听|困惑|疑惑|关切|担心|"
    r"跟着|跟随|看着|盯着|注视|播放音乐|放歌|放音乐|回正|看|动)"
)

_ACTION_WORDS = (
    r"拍照|拍摄|观察|识别|走|挪|移动|前进|后退|左转|右转|转弯|转圈|转头|歪头|低头|"
    r"抬手|举手|挥手|招手|跳舞|点头|跟着|跟随|陪我走|看着|盯着|盯住|注视|开心|"
    r"难过|生气|惊讶|表情|思考|倾听|困惑|疑惑|关切|担心|播放音乐|放歌|放音乐|停止音乐"
)
_THIRD_PARTY_ACTION_RE = re.compile(
    rf"^(?:他|她|它|他们|她们|它们|别人|小[\u4e00-\u9fff]{{1,3}}|那个人).*({_ACTION_WORDS})|"
    rf"^(?!瓦力|你|请|麻烦|帮我|给我)(?:[\u4e00-\u9fff]{{1,4}})(?:正在|在|曾经).*({_ACTION_WORDS})|"
    rf"(?:让|叫|要求)(?!你|瓦力)(?:[\u4e00-\u9fff]{{1,4}}|他|她|它|别人).*({_ACTION_WORDS})"
)
_FIRST_PERSON_ACTION_RE = re.compile(
    rf"^(?:我|我们)(?!.*(?:让你|叫你|要你|请你|命令你|希望你|想看你|"
    rf"不需要你|不用你|不想让你)).*({_ACTION_WORDS})"
)
_QUOTED_ACTION_RE = re.compile(
    rf"[“‘\"'].*(?:{_ACTION_WORDS}).*[”’\"'].*(?:意思|命令句|这句话|那句话|"
    rf"怎么说|怎么读|表示|属于)"
)
_POSTPOSED_NEGATION_RE = re.compile(
    rf"(?:{_ACTION_WORDS}).*(?:不用了?|不要了?|不必|算了|取消|作罢)"
)
_EXPLANATION_CONTEXT_RE = re.compile(
    r"(?:什么意思|什么是|怎么读|如何理解|命令句|这句话|那句话|表示|属于)"
)

_EMOTION_PATTERNS = {
    "curious": r"好奇",
    "happy": r"开心|高兴|快乐",
    "sad": r"难过|伤心|悲伤|沮丧",
    "surprised": r"惊讶|吃惊",
    "disdain": r"鄙视|不屑|白眼",
    "angry": r"生气|愤怒",
}

_SEQUENCE_PATTERNS = {
    "happy_dance": r"跳(?:个|一段)?舞|跳舞|开心舞",
    "wave_hello": r"招手|挥手|挥挥手|打招呼",
    "basic_wave": r"招手|挥手|挥挥手",
    "complex_greet": r"打(?:个)?招呼|问好|问候",
    "basic_nod": r"点(?:一下)?头|点头",
    "sad_react": r"(?:做|来|表现|摆|装).*(?:难过|伤心|悲伤|沮丧).*(?:反应|动作)|"
                 r"(?:难过|伤心|悲伤|沮丧).*(?:反应|动作)",
    "scared": r"(?:做|来|表现|摆|装).*(?:害怕|吓一跳|防御)|"
               r"(?:害怕|吓一跳).*(?:表情|样子|动作)",
    "right_hand_up": r"(?:举|抬)(?:起|一下)?右(?:边)?(?:手|胳膊)|右(?:边)?(?:手|胳膊).*(?:举|抬)",
    "left_hand_up": r"(?:举|抬)(?:起|一下)?左(?:边)?(?:手|胳膊)|左(?:边)?(?:手|胳膊).*(?:举|抬)",
    "raise_hand": r"举(?:起|一下)?手|把手举起来|抬手",
    "arms_up": r"(?:举|抬)(?:起|一下)?(?:双手|两只手)|把(?:双手|两只手)举起来|双手举高|投降",
    "arms_down": r"放下(?:双)?手|把(?:双)?手放下",
    "head_down": r"低(?:一下)?头|把头低下",
    "turn_head_left": r"(?:向|往)?左(?:边)?(?:看|转头|扭头)|看看左边",
    "turn_head_right": r"(?:向|往)?右(?:边)?(?:看|转头|扭头)|看看右边",
    "tilt_head_left": r"(?:向|往)?左(?:边)?(?:歪头|倾头)|头歪向左",
    "tilt_head_right": r"(?:向|往)?右(?:边)?(?:歪头|倾头)|头歪向右",
    "tilt_left": r"(?:向|往)?左(?:边)?(?:歪|倾)|歪向左",
    "tilt_right": r"(?:向|往)?右(?:边)?(?:歪|倾)|歪向右",
    "look_left_up": r"(?:向|往)?左上(?:方)?(?:看|张望)",
    "look_center": r"回正|把头摆正|看向正?前方|目视前方",
    "sad_eyes": r"(?:难过|伤心|悲伤).*(?:眼神|眼睛)",
    "eyebrows_open": r"张开眉毛|抬起眉毛|扬起眉毛",
    "eyebrows_close": r"闭上眉毛|压低眉毛",
    "expression_neutral": r"(?:恢复|回到).*(?:平静|正常|中性)|面无表情",
    "expression_listening": r"(?:认真|专心).*(?:听|倾听)|做.*倾听.*表情",
    "expression_thinking": r"(?:做|来|摆).*(?:思考|想一想|沉思).*(?:表情|样子|动作)?|"
                           r"(?:思考|想一想|沉思).*(?:表情|样子|动作)",
    "expression_happy": r"(?:做|来|摆).*(?:开心|高兴|快乐).*(?:表情|样子|动作)|"
                          r"(?:开心|高兴|快乐).*(?:表情|样子|动作)",
    "expression_sad": r"(?:做|来|摆).*(?:难过|伤心|悲伤|沮丧).*(?:表情|样子|动作)",
    "expression_surprised": r"(?:做|来|摆).*(?:惊讶|吃惊).*(?:表情|样子|动作)|"
                               r"(?:惊讶|吃惊).*(?:表情|样子|动作)",
    "expression_confused": r"(?:做|来|摆).*(?:困惑|疑惑).*(?:表情|样子|动作)|"
                              r"(?:困惑|疑惑).*(?:表情|样子|动作)",
    "expression_concerned": r"(?:做|来|摆).*(?:关心|担心|关切).*(?:表情|样子|动作)|"
                               r"(?:关心|担心|关切).*(?:表情|样子|动作)",
}

_MOVE_PATTERNS = {
    "forward": r"(?:向|往|朝)前(?:走|挪|移动|开|进)|前进",
    "backward": r"后退|倒退|(?:向|往|朝)后(?:走|挪|移动|退)",
    "spin": r"(?:原地)?(?:转|旋转)(?:一|个)?圈|转圈",
    "left": r"左转弯|(?:向|往)左拐|底盘向左|身体向左转|(?:向|往)左(?:走|移动)|"
            r"(?:向)?左转(?:一下)?(?:吧|啊|呀|[。！!]|$)",
    "right": r"右转弯|(?:向|往)右拐|底盘向右|身体向右转|(?:向|往)右(?:走|移动)|"
             r"(?:向)?右转(?:一下)?(?:吧|啊|呀|[。！!]|$)",
}

_TRACKING_PATTERNS = {
    "follow_me": r"跟着我|跟随我|跟我走|跟我来|陪我走|追踪我",
    "look_at_me": r"看着我|看住我|盯着我|盯住我|注视我|看我",
}

# Prefer the most specific, user-facing sequence when multiple configured
# sequences share the same phrase (for example raise_hand/arms_up).  This list
# only translates the consequent action; the visual condition remains entirely
# model-defined and is evaluated later by the vision step.
_CONDITIONAL_SEQUENCE_PRIORITY = (
    "right_hand_up",
    "left_hand_up",
    "basic_nod",
    "raise_hand",
    "arms_up",
    "arms_down",
    "wave_hello",
    "happy_dance",
    "head_down",
    "turn_head_left",
    "turn_head_right",
    "tilt_head_left",
    "tilt_head_right",
    "look_left_up",
    "look_center",
    "sad_react",
    "scared",
    "eyebrows_open",
    "eyebrows_close",
    "expression_neutral",
    "expression_listening",
    "expression_thinking",
    "expression_happy",
    "expression_sad",
    "expression_surprised",
    "expression_confused",
    "expression_concerned",
)

_CONDITIONAL_ACTION_SPLIT_RE = re.compile(r"(?:你?就|然后|则|便)")


def _load_sequence_names():
    path = Path(__file__).resolve().parents[1] / "core" / "sequences.yaml"
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return frozenset()
    return frozenset({
        *map(str, (config.get("sequences") or {}).keys()),
        *map(str, (config.get("poses") or {}).keys()),
    })


_SEQUENCE_NAMES = _load_sequence_names()


def _valid_arguments(name, arguments):
    if not isinstance(arguments, dict):
        return False, "arguments_not_object"

    keys = set(arguments)
    if name == "express_emotion":
        valid = keys == {"emotion"} and arguments.get("emotion") in _EMOTIONS
    elif name == "play_sequence":
        sequence_name = arguments.get("sequence_name")
        valid = (
            keys == {"sequence_name"}
            and isinstance(sequence_name, str)
            and sequence_name in _SEQUENCE_NAMES
        )
    elif name == "move_chassis":
        duration = arguments.get("duration", 1)
        valid = (
            keys <= {"direction", "duration"}
            and "direction" in keys
            and arguments.get("direction") in _DIRECTIONS
            and isinstance(duration, int)
            and not isinstance(duration, bool)
            and 1 <= duration <= 3
        )
    elif name == "set_tracking_mode":
        valid = keys == {"mode"} and arguments.get("mode") in _TRACKING_MODES
    elif name == "set_vision_gate":
        valid = keys == {"enabled"} and isinstance(arguments.get("enabled"), bool)
    elif name == "inspect_camera":
        question = arguments.get("question", "")
        valid = (
            keys <= {"question"}
            and isinstance(question, str)
            and len(question) <= 500
        )
    elif name == "control_music":
        action = arguments.get("action")
        track = arguments.get("track", "")
        valid = (
            keys <= {"action", "track"}
            and action in _MUSIC_ACTIONS
            and isinstance(track, str)
            and len(track) <= 200
            and "/" not in track
            and "\\" not in track
            and (action == "play" or not track)
        )
    elif name == "stop_all":
        valid = not keys
    elif name == "run_conditional_task":
        try:
            from services.conditional_task import normalize_conditional_task_plan

            normalize_conditional_task_plan(arguments)
            valid = True
        except (TypeError, ValueError):
            valid = False
    else:
        return False, "unknown_tool"
    return (True, "") if valid else (False, "invalid_arguments")


def _obvious_non_command(user_text):
    compact = "".join(str(user_text or "").split())
    if re.fullmatch(r"(?:你好|您好|嗨|哈喽|早上好|中午好|晚上好)[！!。.]?", compact):
        return True
    if re.search(r"(?:没有|没|并未)(?:让|叫|要求|命令)", compact):
        return True
    if re.search(r"(?:故事|什么意思|什么是|怎么读|如何理解|为什么)", compact):
        return True
    if re.search(r"(?:昨天|昨晚|上次|刚刚|刚才|之前|曾经|已经)", compact) and not re.search(
        r"(?:今天|现在|马上|立刻|接着|然后|再)", compact
    ):
        return True
    if re.search(r"(?:如果|假如|要是)", compact) and not _REQUEST_CUE_RE.search(compact):
        return True
    if _THIRD_PARTY_ACTION_RE.search(compact):
        return True
    if _FIRST_PERSON_ACTION_RE.search(compact):
        return True
    if _QUOTED_ACTION_RE.search(compact):
        return True
    return bool(
        _CAPABILITY_QUESTION_RE.search(compact)
        and not _REQUEST_CUE_RE.search(compact)
    )


def _requested_duration(compact):
    match = re.search(r"([一二两三123])秒", compact)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return {"一": 1, "二": 2, "两": 2, "三": 3}[token]


def _matches_sequence_intent(compact, sequence_name):
    pattern = _SEQUENCE_PATTERNS.get(sequence_name)
    if not pattern:
        return False
    if sequence_name in {"turn_head_left", "turn_head_right"}:
        left = bool(re.search(r"左(?:边)?(?:看|转头|扭头)|看看左边", compact))
        right = bool(re.search(r"右(?:边)?(?:看|转头|扭头)|看看右边", compact))
        if left or right:
            return left if sequence_name == "turn_head_left" else right
        return bool(re.search(r"(?:转|旋转)(?:个|一下)?头|转头", compact))
    return bool(re.search(pattern, compact))


def _conditional_action_clause(user_text):
    compact = "".join(str(user_text or "").split())
    return _CONDITIONAL_ACTION_SPLIT_RE.split(compact)[-1]


def _explicit_conditional_action(action_clause):
    """Translate an explicit consequent into one allowlisted action.

    Return ``None`` for unfamiliar wording so a valid model proposal can still
    be used.  The rules intentionally inspect only the text after 就/然后/则/便;
    an object or gesture mentioned in the condition can never become an action.
    """
    if not action_clause:
        return None

    if _TRACKING_STOP_RE.search(action_clause) or _LEADING_TRACKING_STOP_RE.search(
        action_clause
    ):
        return "set_tracking_mode", {"mode": "idle"}
    if _VISION_OFF_RE.search(action_clause) or _LEADING_VISION_OFF_RE.search(
        action_clause
    ):
        return "set_vision_gate", {"enabled": False}
    if re.search(r"(?:打开|开启|启动|启用).{0,8}(?:视觉|跟踪)", action_clause):
        return "set_vision_gate", {"enabled": True}

    tracking_modes = [
        mode
        for mode, pattern in _TRACKING_PATTERNS.items()
        if re.search(pattern, action_clause)
    ]
    if tracking_modes:
        return "set_tracking_mode", {"mode": tracking_modes[0]}

    for sequence_name in _CONDITIONAL_SEQUENCE_PRIORITY:
        if sequence_name in _SEQUENCE_NAMES and _matches_sequence_intent(
            action_clause, sequence_name
        ):
            return "play_sequence", {"sequence_name": sequence_name}

    emotions = [
        emotion
        for emotion, pattern in _EMOTION_PATTERNS.items()
        if re.search(pattern, action_clause)
    ]
    if emotions:
        return "express_emotion", {"emotion": emotions[0]}

    if re.search(
        r"(?:停止|停下|停住)(?:所有|全部)?(?:动作|运动)|"
        r"(?:所有|全部)(?:动作|运动)?(?:停止|停下)|紧急停止|别动|不要动",
        action_clause,
    ):
        return "stop_all", {}
    return None


def canonicalize_conditional_action(user_text, plan):
    """Correct a model plan from the user's explicit consequent, if known.

    This makes common robot actions portable across models without hard-coding
    what the camera is supposed to see.  Unknown actions are left untouched and
    must still pass the ordinary runtime allowlist.
    """
    if not isinstance(plan, dict):
        return plan
    explicit = _explicit_conditional_action(_conditional_action_clause(user_text))
    if explicit is None:
        return dict(plan)
    action_name, action_arguments = explicit
    normalized = dict(plan)
    normalized["action_name"] = action_name
    normalized["action_arguments"] = action_arguments
    return normalized


def _has_explicit_argument_conflict(compact, name, arguments):
    """Return True only when text clearly contradicts proposed arguments.

    Absence of a hard-coded synonym is not a conflict: the model has already
    performed the semantic selection.  Keeping this check conflict-based makes
    colloquial and mildly corrupted ASR text usable without giving up concrete
    direction, duration, mode, or polarity checks.
    """
    if name == "move_chassis":
        mentioned = {
            direction
            for direction, pattern in _MOVE_PATTERNS.items()
            if re.search(pattern, compact)
        }
        if mentioned and arguments["direction"] not in mentioned:
            return True
        duration = _requested_duration(compact)
        proposed_duration = arguments.get("duration", 1)
        return proposed_duration != 1 if duration is None else duration != proposed_duration
    if name == "play_sequence":
        mentioned = {
            sequence_name
            for sequence_name in _SEQUENCE_PATTERNS
            if _matches_sequence_intent(compact, sequence_name)
        }
        return bool(mentioned and arguments["sequence_name"] not in mentioned)
    if name == "set_tracking_mode":
        mentioned = {
            mode
            for mode, pattern in _TRACKING_PATTERNS.items()
            if re.search(pattern, compact)
        }
        return bool(mentioned and arguments["mode"] not in mentioned)
    if name == "set_vision_gate":
        opening = (
            r"(?:打开|开启|启动|启用).*(?:视觉|跟踪)|"
            r"(?:视觉|跟踪).*(?:打开|开启|启动|启用)"
        )
        closing = (
            r"(?:关闭|关掉|停止|退出).*(?:视觉|跟踪)|"
            r"(?:视觉|跟踪).*(?:关闭|关掉|停止|退出)"
        )
        explicitly_open = bool(re.search(opening, compact))
        explicitly_closed = bool(re.search(closing, compact))
        return (
            (explicitly_open and arguments["enabled"] is False)
            or (explicitly_closed and arguments["enabled"] is True)
        )
    if name == "express_emotion":
        mentioned = {
            emotion
            for emotion, pattern in _EMOTION_PATTERNS.items()
            if re.search(pattern, compact)
        }
        return bool(mentioned and arguments["emotion"] not in mentioned)
    return False


def validate_action_call(user_text, name, arguments):
    """Return ``(allowed, reason)`` for one LLM-proposed action call."""
    valid, reason = _valid_arguments(name, arguments)
    if not valid:
        return False, reason

    compact = "".join(str(user_text or "").split())
    if name == "run_conditional_task":
        from services.conditional_task import (
            is_conditional_task_request,
            normalize_conditional_task_plan,
        )

        if not is_conditional_task_request(compact):
            return False, "not_conditional_command"
        plan = normalize_conditional_task_plan(arguments)
        action_clause = _conditional_action_clause(compact)
        stop_action = (
            plan["action_name"] == "stop_all"
            or (
                plan["action_name"] == "set_tracking_mode"
                and plan["action_arguments"].get("mode") == "idle"
            )
            or (
                plan["action_name"] == "set_vision_gate"
                and plan["action_arguments"].get("enabled") is False
            )
        )
        if (
            _EXPLANATION_CONTEXT_RE.search(compact)
            or _QUOTED_ACTION_RE.search(compact)
            or _CANCEL_STOP_RE.search(compact)
            or _CANCEL_EXPLICIT_STOP_RE.search(compact)
            or (
                not stop_action
                and (
                    _NEGATED_ACTION_RE.search(action_clause)
                    or _POSTPOSED_NEGATION_RE.search(action_clause)
                )
            )
        ):
            return False, "non_command_context"
        explicit_action = _explicit_conditional_action(action_clause)
        if explicit_action is not None and (
            plan["action_name"], plan["action_arguments"]
        ) != explicit_action:
            return False, "argument_conflict"
        if _has_explicit_argument_conflict(
            action_clause, plan["action_name"], plan["action_arguments"]
        ):
            return False, "argument_conflict"
        return True, ""

    if _CANCEL_STOP_RE.search(compact) or _CANCEL_EXPLICIT_STOP_RE.search(compact):
        return False, "negated_action"
    if _LEADING_TRACKING_STOP_RE.search(compact) and not _EXPLANATION_CONTEXT_RE.search(compact):
        if name == "set_tracking_mode" and arguments.get("mode") == "idle":
            return True, ""
        return False, "stop_command_mismatch"
    if _LEADING_VISION_OFF_RE.search(compact) and not _EXPLANATION_CONTEXT_RE.search(compact):
        if name == "set_vision_gate" and arguments.get("enabled") is False:
            return True, ""
        return False, "stop_command_mismatch"
    if _obvious_non_command(compact):
        return False, "non_command_context"
    if _TRACKING_STOP_RE.search(compact):
        if name == "set_tracking_mode" and arguments.get("mode") == "idle":
            return True, ""
        return False, "stop_command_mismatch"
    if _VISION_OFF_RE.search(compact):
        if name == "set_vision_gate" and arguments.get("enabled") is False:
            return True, ""
        return False, "stop_command_mismatch"
    music_stop = bool(re.search(
        r"(?:停止|停下|关闭|关掉|别再|不要再).*(?:音乐|歌曲|歌)|"
        r"(?:音乐|歌曲|歌).*(?:停止|停下|关闭|关掉)", compact
    ))
    if music_stop:
        if name == "control_music" and arguments.get("action") == "stop":
            return True, ""
        return False, "stop_command_mismatch"
    if name == "control_music" and arguments.get("action") == "stop":
        if re.search(r"(?:播放|放).*(?:音乐|歌曲|歌)", compact):
            return False, "argument_conflict"
    # “盯住我别乱看” negates wandering, not the requested tracking action.
    negation_text = compact.replace("别乱看", "").replace("不要乱看", "")
    if _POSTPOSED_NEGATION_RE.search(negation_text):
        return False, "negated_action"
    if _NEGATED_ACTION_RE.search(negation_text):
        return False, "negated_action"
    if _has_explicit_argument_conflict(compact, name, arguments):
        return False, "argument_conflict"
    return True, ""


def deterministic_safety_action(user_text):
    """Return a local fail-safe action for an explicit tracking stop command.

    Stopping an active visual-follow mode must not depend on a model producing
    a tool call.  Keep this deliberately narrow: explanations, narration, and
    commands that negate the stop itself continue through normal conversation.
    """
    compact = "".join(str(user_text or "").split())
    if not compact:
        return None
    if (
        _EXPLANATION_CONTEXT_RE.search(compact)
        or _CANCEL_STOP_RE.search(compact)
        or _CANCEL_EXPLICIT_STOP_RE.search(compact)
        or _obvious_non_command(compact)
    ):
        return None
    if _LEADING_TRACKING_STOP_RE.search(compact) or _TRACKING_STOP_RE.search(compact):
        return "set_tracking_mode", {"mode": "idle"}
    return None


def validate_action_arguments(name, arguments):
    """Validate an already-authorized structured action request.

    This is intentionally narrower than :func:`validate_action_call`: an MCP
    server must not trust client-supplied natural-language evidence. Identity,
    approval, and rate limits belong at the transport/gateway boundary, while
    this function enforces the robot's argument allowlist and physical limits.
    """
    return _valid_arguments(name, arguments)


def registered_sequence_names():
    """Return the immutable configured sequence/pose allowlist."""
    return tuple(sorted(_SEQUENCE_NAMES))
