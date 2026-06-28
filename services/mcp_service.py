# services/mcp_service.py
# 瓦力工具注册中心 — 纯签名声明
# ROS 模式下仅负责告诉 LLM "有哪些工具可用"，具体执行由各 ROS 节点完成
# LLM 返回 tool_call → llm_ros_node 发到 /action_cmd → 对应节点执行

import asyncio
import os
import yaml
from fastmcp import FastMCP

mcp = FastMCP("Wali_Action_Center")

# 已知动作的中英文语义映射字典（用于增强大模型的语义理解）
_semantic_mappings = {
    "happy_dance": "开心跳舞转圈",
    "wave_hello": "招手/打招呼",
    "sad_react": "难过反应/低迷",
    "scared": "害怕吓一跳/防御",
    "raise_hand": "举手/引起注意",
    "basic_nod": "点头肯定/同意",
    "basic_wave": "简单的单手挥动",
    "arms_up": "举手/抬手/双手举高/投降",
    "arms_down": "放下双手",
    "head_down": "低头/沮丧",
    "turn_head_left": "向左看/左转头",
    "turn_head_right": "向右看/右转头",
    "tilt_head_left": "向左歪头/左倾/左眼下右眼上",
    "tilt_head_right": "向右歪头/右倾/右眼下左眼上",
    "look_left_up": "左上张望/思考",
    "look_center": "回正/往前看"
}

# 动态读取动作编排文件，生成动作菜单
def _build_sequence_prompt():
    base_prompt = (
        "控制瓦力的物理躯体做出各种动作。这是你控制身体动作的唯一指定工具！\n\n"
        "【核心智能要求】：你应当具备语义意图识别能力！当用户的要求（例如“抬手”、“伸个手”、“向右看”）"
        "与下方列表并非字面完全一致时，你必须自己理解意图，并选择一个最接近的动作调用，绝对不要因为字面不一致就拒绝调用工具！\n\n"
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
# 表情 / 动作类（躯干表演，由 servo_ros_node 执行）
# ==========================================

@mcp.tool()
def express_emotion(emotion: str) -> str:
    """
    控制瓦力表达情绪。
    
    emotion 可选：
      - "curious"  : 好奇，眼睛微动
      - "happy"    : 开心，眉毛上扬、欢快动作
      - "sad"      : 难过，手部低垂、眼睛低落
      - "surprised": 惊讶，眼睛瞪大、眉毛上扬、脖子后仰
      - "disdain"  : 鄙视/翻白眼
      - "angry"    : 生气
    
    通过 ROS /action_cmd 下发，由 servo_ros_node 执行。
    """
    return "ok"


@mcp.tool(description=_play_sequence_doc)
def play_sequence(sequence_name: str) -> str:
    return "ok"


# ==========================================
# 底盘类（由 motor_ros_node 执行）
# ==========================================

@mcp.tool()
def move_chassis(direction: str, duration: int = 1) -> str:
    """
    控制瓦力履带底盘移动。
    
    【警告】：如果用户只是让你“向左看”、“向右看”或者“转头”，请调用 play_sequence 工具！只有当用户明确要求“走动”、“移动身体”、“转身”、“前进后退”时，才使用本底盘控制工具！
    
    direction 可选：
      - "forward"  : 前进
      - "backward" : 后退
      - "spin"     : 原地转圈
      - "left"     : 左转弯
      - "right"    : 右转弯
    
    duration: 持续秒数，默认 1 秒，建议 1~3 秒。
    
    通过 ROS /action_cmd 下发，由 motor_ros_node 执行。
    """
    return "ok"


# ==========================================
# 视觉跟踪类（由 wali_tracking_node 执行）
# ==========================================

@mcp.tool()
def set_tracking_mode(mode: str) -> str:
    """
    切换瓦力视觉跟踪模式。
    参数 mode:
      "body_follow" : 人体跟随，底盘保持人在画面中央并控制距离
      "face_follow" : 人脸跟随，脖子俯仰跟踪 + 底盘辅助
      "idle"        : 退出跟踪，底盘停止
    通过 ROS /action_cmd 下发，由 wali_tracking_node 执行。
    """
    return "ok"


@mcp.tool()
def set_vision_gate(enabled: bool) -> str:
    """
    打开或关闭视觉跟踪。
    enabled=True 默认进入 body_follow，False 退出所有跟踪。
    通过 ROS /action_cmd 下发，由 wali_tracking_node 执行。
    """
    return "ok"


# ==========================================
# 桥接接口（供 llm_service.py 调用）
# ==========================================

def get_chat_tools():
    """将 FastMCP 注册的工具列表转换成 LLM function calling 格式"""
    tools = []
    try:
        mcp_tools = asyncio.run(mcp.list_tools())
        for tool in mcp_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
    except Exception as e:
        print(f"[MCP] 获取工具列表失败: {e}")
    return tools


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