# services/mcp_service.py
# 瓦力工具注册中心 — 纯签名声明
# ROS 模式下仅负责告诉 LLM "有哪些工具可用"，具体执行由各 ROS 节点完成
# LLM 返回 tool_call → llm_ros_node 发到 /action_cmd → 对应节点执行

import asyncio
import copy
import logging
import os
from typing import Any
import yaml
from fastmcp import FastMCP

from services.conditional_task import (
    CONDITIONAL_TASK_TOOL_NAME,
    conditional_task_tool_schema,
)

mcp = FastMCP("Wali_Action_Center")
LOGGER = logging.getLogger(__name__)


class MCPToolDiscoveryError(RuntimeError):
    """Raised when function tools cannot be supplied to an LLM request."""


_ACTION_TOOL_BOUNDARY = (
    "仅当用户明确命令瓦力现在执行该动作时调用。能力询问、疑问、假设、故事、引用、"
    "词义解释、过去事件、第三方行为或单纯提到动作时禁止调用；不确定时直接回答或澄清。"
)

# 已知动作的中英文语义映射字典（用于增强大模型的语义理解）
_semantic_mappings = {
    "happy_dance": "开心跳舞转圈",
    "wave_hello": "招手/打招呼",
    "sad_react": "难过反应/低迷",
    "scared": "害怕吓一跳/防御",
    "raise_hand": "举右手示意/引起注意（组合动作）",
    "basic_nod": "点头肯定/同意",
    "basic_wave": "简单的单手挥动",
    "right_hand_up": "只举右手（基础动作）",
    "left_hand_up": "只举左手（基础动作）",
    "arms_up": "双手举高/投降（基础动作）",
    "arms_down": "放下双手",
    "head_down": "低头/沮丧",
    "turn_head_left": "向左看/左转头",
    "turn_head_right": "向右看/右转头",
    "tilt_head_left": "向左歪头/左倾/左眼下右眼上",
    "tilt_head_right": "向右歪头/右倾/右眼下左眼上",
    "look_left_up": "左上张望/思考",
    "look_center": "回正/往前看",
    "expression_neutral": "平静待机/回到中性表情",
    "expression_listening": "认真倾听/注意力集中",
    "expression_thinking": "思考/回忆/斟酌",
    "expression_happy": "开心/高兴",
    "expression_sad": "难过/低落",
    "expression_surprised": "惊讶/吃惊，伸出脖子",
    "expression_confused": "疑惑/困惑",
    "expression_concerned": "关切/担心",
}

# 动态读取动作编排文件，生成动作菜单
def _build_sequence_prompt():
    base_prompt = (
        f"{_ACTION_TOOL_BOUNDARY}\n"
        "控制瓦力的头、手臂或身体做一次预设表演动作。只在明确动作命令中，根据语义"
        "选择最接近的预设；向左/右看属于转头，不是移动底盘。\n\n"
        "sequence_name 必须是以下预设动作之一：\n"
    )
    
    try:
        yaml_path = os.path.join(os.path.dirname(__file__), '../core/sequences.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            seq_data = yaml.safe_load(f) or {}
            
        seqs = list(seq_data.get('sequences', {}).keys())
        poses = list(seq_data.get('poses', {}).keys())
        
        # 加上中文语义后缀
        seqs_with_semantics = [f"{s}({_semantic_mappings[s]})" if s in _semantic_mappings else s for s in seqs]
        poses_with_semantics = [f"{p}({_semantic_mappings[p]})" if p in _semantic_mappings else p for p in poses]
        
        menu = []
        if seqs:
            menu.append("【成组复杂剧本 (Sequences)】: " + ", ".join(seqs_with_semantics))
        if poses:
            menu.append("【基础单点动作 (Poses)】: " + ", ".join(poses_with_semantics))
            
        return base_prompt + "\n".join(menu)
    except Exception as e:
        print(f"[MCP] 读取 sequences.yaml 失败: {e}")
        return base_prompt + "wave_hello, happy_dance, sad_react, scared, basic_nod, arms_up, turn_head_left"

_play_sequence_doc = _build_sequence_prompt()


# ==========================================
# 表情 / 动作类（躯干表演，由 sequence_ros_node 执行）
# ==========================================

@mcp.tool()
def express_emotion(emotion: str) -> str:
    """
    仅当用户明确命令瓦力现在表达某种情绪时，控制瓦力用身体表达情绪。
    询问瓦力是否开心、讨论情绪或普通闲聊时不能为了生动而自行调用。
    
    emotion 可选：
      - "curious"  : 好奇，眼睛微动
      - "happy"    : 开心，眉毛上扬、欢快动作
      - "sad"      : 难过，手部低垂、眼睛低落
      - "surprised": 惊讶，眼睛瞪大、眉毛上扬、脖子后仰
      - "disdain"  : 鄙视/翻白眼
      - "angry"    : 生气
    
    通过 ROS /action_cmd 下发，由 sequence_ros_node 执行。
    """
    return "ok"


@mcp.tool(description=_play_sequence_doc)
def play_sequence(sequence_name: str) -> str:
    return "ok"


# ==========================================
# 底盘类（由 sequence_ros_node 分发到所选硬件后端）
# ==========================================

@mcp.tool()
def move_chassis(direction: str, duration: int = 1) -> str:
    """
    仅当用户明确命令瓦力现在移动时，控制瓦力履带底盘短距离移动。能力询问、
    疑问、假设、故事、过去事件或单纯提到移动时禁止调用。
    
    【警告】：如果用户只是让你“向左看”、“向右看”或者“转头”，请调用 play_sequence 工具！只有当用户明确要求“走动”、“移动身体”、“转身”、“前进后退”时，才使用本底盘控制工具！
    
    direction 可选：
      - "forward"  : 前进
      - "backward" : 后退
      - "spin"     : 原地转圈
      - "left"     : 左转弯
      - "right"    : 右转弯
    
    duration: 持续秒数，只允许 1~3 秒，默认 1 秒。
    
    通过 ROS /action_cmd 下发，由 sequence_ros_node 执行。
    """
    return "ok"


# ==========================================
# 视觉跟踪类（由 wali_tracking_node 执行）
# ==========================================

@mcp.tool()
def set_tracking_mode(mode: str) -> str:
    """
    仅当用户明确命令瓦力现在切换持续视觉跟踪模式时调用。仅仅询问、引用、
    解释或提到“看着我/跟着我”等短语不能调用。

      - 明确命令“看着我/盯着我/注视我”时传入 mode="look_at_me"
      - 明确命令“跟着我/跟随我”时传入 mode="follow_me"
      - 明确命令“别看了/停止跟随/退出跟踪”时传入 mode="idle"
      
    参数 mode 可选值:
      "follow_me"  : 人体跟随，底盘保持人在画面中央并控制距离
      "look_at_me" : 视觉注视，脑袋和眼睛死死锁定并看着用户
      "idle"       : 退出视觉跟踪
      
    注意：这是持续性的 AI 视觉锁定模式，非一次性动作。
    """
    return "ok"


@mcp.tool()
def set_vision_gate(enabled: bool) -> str:
    """
    仅当用户明确命令瓦力现在打开或关闭视觉跟踪总开关时调用。
    enabled=True 默认进入 body_follow，False 退出所有跟踪。
    通过 ROS /action_cmd 下发，由 wali_tracking_node 执行。
    """
    return "ok"


@mcp.tool()
def inspect_camera(question: str = "") -> str:
    """仅当用户明确要求瓦力现在观察、拍摄或识别眼前现实画面时调用。

    拍摄一帧当前摄像头画面并回答用户的视觉问题。谈论视觉或询问瓦力是否能看见
    不等于要求立即拍摄。系统会自动抓取一帧画面并交给视觉模型，question 应保留
    用户想知道的内容。不要把“看着我/跟着我”当成此工具，那些属于持续视觉跟随，
    应使用 set_tracking_mode。
    """
    return "ok"


@mcp.tool()
def run_conditional_task(
    observation: str,
    condition: str,
    action_name: str,
    action_arguments: dict[str, Any],
) -> str:
    """执行一次“观察画面 → 判断条件 → 条件成立才动作”的复合任务。

    仅当用户明确要求瓦力现在观察真实环境，并根据观察结果决定是否执行一个动作时调用。
    不要把它拆成 inspect_camera 和独立动作工具，也不要同时调用本工具与 action_name 对应的
    动作工具。observation 描述要观察什么；condition 是仅依据当前画面判断的完整条件，
    可以是任意物体、人物、颜色、姿态、数量或空间关系，不要写死特定目标；action_name 和
    action_arguments 描述条件明确成立时执行的一个动作。举手、点头、挥手、转头等预设动作
    必须使用 action_name="play_sequence"，例如举手参数为 {"sequence_name":"raise_hand"}、
    点头为 {"sequence_name":"basic_nod"}。条件不成立或无法确定时不会动作。
    能力询问、举例、假设讨论、故事、引用或没有要求立即执行的句子禁止调用。
    """
    return "ok"


# ==========================================
# 桥接接口（供 llm_service.py 调用）
# ==========================================


def _configured_sequence_names():
    yaml_path = os.path.join(os.path.dirname(__file__), '../core/sequences.yaml')
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return []
    return [
        *map(str, (config.get('sequences') or {}).keys()),
        *map(str, (config.get('poses') or {}).keys()),
    ]


def _tighten_tool_schema(name, parameters):
    """Add provider-visible constraints; runtime validation remains mandatory."""
    if name == CONDITIONAL_TASK_TOOL_NAME:
        return conditional_task_tool_schema()
    schema = copy.deepcopy(parameters)
    schema['additionalProperties'] = False
    properties = schema.setdefault('properties', {})
    if name == 'express_emotion' and 'emotion' in properties:
        properties['emotion']['enum'] = sorted({
            'curious', 'happy', 'sad', 'surprised', 'disdain', 'angry'
        })
    elif name == 'play_sequence' and 'sequence_name' in properties:
        sequence_names = _configured_sequence_names()
        if sequence_names:
            properties['sequence_name']['enum'] = sequence_names
    elif name == 'move_chassis':
        if 'direction' in properties:
            properties['direction']['enum'] = [
                'forward', 'backward', 'spin', 'left', 'right'
            ]
        if 'duration' in properties:
            properties['duration'].update({'minimum': 1, 'maximum': 3})
    elif name == 'set_tracking_mode' and 'mode' in properties:
        properties['mode']['enum'] = ['follow_me', 'look_at_me', 'idle']
    elif name == 'inspect_camera' and 'question' in properties:
        properties['question']['maxLength'] = 500
    return schema

def get_chat_tools():
    """Return FastMCP 2.x tools as OpenAI function-calling declarations.

    FastMCP 2.x exposes ``get_tools()`` (a mapping), not the old
    ``list_tools()`` API.  An empty or malformed result is an operational
    failure: silently returning ``[]`` would make the model explain actions in
    text instead of being able to call the robot-control functions.
    """
    try:
        getter = getattr(mcp, "get_tools", None)
        if not callable(getter):
            # Kept only for an explicitly older FastMCP installation. The
            # project requirement pins FastMCP >=2,<3, where get_tools is used.
            getter = getattr(mcp, "list_tools", None)
        if not callable(getter):
            raise MCPToolDiscoveryError("FastMCP 缺少 get_tools()；项目要求 FastMCP 2.x")
        registered = asyncio.run(getter())
        raw_tools = registered.values() if isinstance(registered, dict) else registered
        tools = []
        for tool in raw_tools:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            parameters = getattr(tool, "parameters", None)
            if not isinstance(name, str) or not name:
                raise MCPToolDiscoveryError("FastMCP 返回了没有名称的工具")
            if not isinstance(description, str):
                raise MCPToolDiscoveryError(f"FastMCP 工具 {name} 没有有效说明")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise MCPToolDiscoveryError(f"FastMCP 工具 {name} 没有合法 JSON Schema")
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": _tighten_tool_schema(name, parameters),
                },
            })
        if not tools:
            raise MCPToolDiscoveryError("FastMCP 未枚举到任何控制工具")
        return tools
    except MCPToolDiscoveryError as exc:
        LOGGER.error("FastMCP control-tool configuration error: %s", exc)
        raise
    except Exception as exc:
        error = MCPToolDiscoveryError(f"FastMCP 2.x 工具枚举失败: {exc}")
        LOGGER.exception("FastMCP control-tool enumeration failed")
        raise error from exc


def execute_tool(name, args_json):
    """
    旧 test.py 桥接接口，ROS 模式下已不走此路径。
    保留以兼容现有测试脚本。
    """
    import json
    args = json.loads(args_json)
    try:
        result = asyncio.run(mcp.call_tool(name, arguments=args))
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {e}"
