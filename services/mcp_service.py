# services/mcp_service.py
import time
from fastmcp import FastMCP
#from services.servo_control import ServoControl
#from services.serial_bridge import SerialBridge

# [ZH] 创建 FastMCP 实例。它会自动将标记了 @mcp.tool() 的函数暴露给大模型。
# [EN] Create a FastMCP instance. It automatically exposes functions marked with @mcp.tool() to the LLM.
mcp = FastMCP("Wali_Action_Center")

# [ZH] 初始化底层驱动层 (肌肉层)
# [EN] Initialize the low-level driver layer (Muscle layer)
#servo = ServoControl()
#serial = SerialBridge()

@mcp.tool()
def express_emotion(emotion: str) -> str:
    """
    [ZH] 控制瓦力表达情绪。支持：curious(好奇), happy(开心), sad(难过), surprised(惊讶)。
    [EN] Control Wali to express emotions. Supports: curious, happy, sad, surprised.
    """
    if emotion == "curious":
        # [ZH] 好奇：歪头，眉毛一高一低
        # [EN] Curious: Tilt head, eyebrows at different heights
        print("执行了动作好奇")
    
    elif emotion == "happy":
        # [ZH] 开心：挥手，眉毛上扬，原地小转
        # [EN] Happy: Wave arms, eyebrows up, small spin
        print("执行了动作开心")
        
    elif emotion == "sad":
        # [ZH] 难过：垂头丧气，手臂放下
        # [EN] Sad: Droop head, lower arms
        print("执行了动作难过")
        
    elif emotion == "surprised":
        # [ZH] 惊讶：身体后仰，眼睛瞪大
        # [EN] Surprised: Lean back, eyes wide open
        print("执行了动作惊讶")
        
    return f"Emotion {emotion} executed."

@mcp.tool()
def perform_action(action: str) -> str:
    """
    [ZH] 执行特定行为。支持：dance(跳舞), talk_micro_move(对话微动)。
    [EN] Perform specific actions. Supports: dance, talk_micro_move.
    """
    if action == "dance":
        # [ZH] 简单的舞蹈逻辑：左右摇摆
        # [EN] Simple dance logic: Sway left and right
        print("执行了动作跳舞")
        
    elif action == "talk_micro_move":
        # [ZH] 对话微动：手部和头部小幅度随机摆动，让瓦力看起来在思考或倾听
        # [EN] Talk micro-move: Subtle random movements for hands and head to make Wali look alive.
        print("说话微微动")
        
    return f"Action {action} finished."

@mcp.tool()
def move_chassis(direction: str, duration: float = 1.0) -> str:
    """
    [ZH] 控制履带底盘移动。支持：forward(前进), backward(后退), spin(旋转)。
    [EN] Control the chassis. Supports: forward, backward, spin.
    """
    # [ZH] 通过串口网桥发送指令给底盘 ESP32
    # [EN] Send command to chassis ESP32 via serial bridge
    cmd = f"move:{direction}"
    #serial.send_command(cmd)
    
    time.sleep(duration)
    
    #serial.send_command("move:stop")
    return f"Moved {direction} for {duration}s."

# ==========================================
# 桥接与兼容层 (供外层 LLM 和 main 调用)
# ==========================================
import asyncio # 🚀 确保在这附近或者文件顶部引入了 asyncio

def execute_tool(name, args_json):
    """供 test.py 调用的接口，解析 JSON 并交给 FastMCP 执行"""
    import json
    args = json.loads(args_json)
    try:
        # 🚀 修复点：使用 asyncio.run() 将异步的 call_tool 转换为同步执行
        result = asyncio.run(mcp.call_tool(name, arguments=args))
        
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


def get_chat_tools():
    """供 llm_service.py 调用的接口，把 FastMCP 的工具翻译成大模型认识的 JSON"""
    tools = []
    try:
        mcp_tools = asyncio.run(mcp.list_tools())
        
        for tool in mcp_tools: 
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    
                    # 🚀 修复点：直接使用 fastmcp 封装好的 parameters 属性！
                    "parameters": tool.parameters 
                }
            })
    except Exception as e:
        print(f"⚠️ 获取工具列表失败: {e}")
    return tools
