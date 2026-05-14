# services/skills.py

WALI_SKILLS = [
    {
        "type": "function",
        "function": {
            "name": "express_emotion",
            "description": "控制瓦力表达丰富的情绪。调用此工具后，底层的眉毛、眼睛、脖子及四肢会自动进行复杂的协同动作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": ["curious", "happy", "sad", "surprised"],
                        "description": "情绪类型：\n"
                                       "- curious(好奇): 左右眼睛及眉毛微动，展现探索欲。\n"
                                       "- happy(开心): 所有舵机加履带电机欢快动作，眉毛上扬。\n"
                                       "- sad(难过): 手部低垂，眼睛低落。\n"
                                       "- surprised(惊讶): 左右眼睛瞪大，眉毛上扬，脖子后仰。"
                    }
                },
                "required": ["emotion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_action",
            "description": "控制瓦力执行特定的行为动作或交互微动。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["dance", "talk_micro_move"],
                        "description": "动作类型：\n"
                                       "- dance(跳舞): 伴随全身所有舵机加电机的复杂舞蹈动作。\n"
                                       "- talk_micro_move(对话微动): 手部舵机和眼部舵机进行小幅度的自然配合，适合在长段对话或打招呼时调用，增加真实感。"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_chassis",
            "description": "控制瓦力的履带底盘进行空间位置的移动。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "spin"],
                        "description": "移动方向：forward(前进), backward(后退), spin(原地转圈)"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "动作持续的秒数（推荐 1 到 3 秒）。",
                        "default": 1
                    }
                },
                "required": ["direction"]
            }
        }
    }
]