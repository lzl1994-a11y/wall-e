# services/mcp_service.py
# 瓦力工具注册中心 — 纯签名声明
# ROS 模式下仅负责告诉 LLM "有哪些工具可用"，具体执行由各 ROS 节点完成
# LLM 返回 tool_call → llm_ros_node 发到 /action_cmd → 对应节点执行

import asyncio
from fastmcp import FastMCP

mcp = FastMCP("Wali_Action_Center")


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


@mcp.tool()
def perform_action(action: str) -> str:
    """
    执行特定行为动作。
    
    action 可选：
      - "dance"            : 全身跳舞
      - "talk_micro_move"  : 对话微动（手+眼小幅自然动作）
      - "wave"             : 挥手
      - "nod"              : 点头
      - "shake_head"       : 摇头
      - "look_up"          : 抬头
      - "look_down"        : 低头
      - "tilt_head"        : 歪头
    
    通过 ROS /action_cmd 下发，由 servo_ros_node 执行。
    """
    return "ok"


# ==========================================
# 底盘类（由 motor_ros_node 执行）
# ==========================================

@mcp.tool()
def move_chassis(direction: str, duration: int = 1) -> str:
    """
    控制瓦力履带底盘移动。
    
    direction 可选：
      - "forward"  : 前进
      - "backward" : 后退
      - "spin"     : 原地转圈
      - "left"     : 左转
      - "right"    : 右转
    
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