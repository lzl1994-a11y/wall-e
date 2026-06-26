import json
import yaml
from openai import OpenAI
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

with open('core/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

client = OpenAI(
    api_key=config['llm']['key'],
    base_url=config['llm']['url']
)

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'move_chassis',
            'description': 'Move the chassis',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {'type': 'string'},
                    'duration': {'type': 'integer'}
                },
                'required': ['direction', 'duration']
            }
        }
    }
]

system_prompt = """
  1. **彻底的物理沉浸：** 你就是一个真实的机器人！在任何情况下，【绝对不允许】在回复中提到"大模型"、"AI"、"代码"、"工具(Tools)"、"舵机(Servo)"、"TFT屏幕"、"内部处理"等词汇。
  2. **拒绝做哑巴：** 只要用户与你互动，你【必须】同时进行两项操作：生成符合毒舌人设的文字回复 + 调用后台提供给你的躯体控制工具。如果你需要做动作，你必须使用提供的 Function Calling 机制调用工具，不要在文本里假装调用。
"""

augmented_prompt = (
    f"【原始语音识别文本】: \"向前走\"\n"
    f"【核心指令与输出规范】\n"
    f"1. 你的回复的第一行必须是：【修正文本】: [纠正后的文本]\n"
    f"2. 从第二行开始，输出你的对话回复。如果需要移动，请务必触发 move_chassis 函数调用！\n\n"
    f"请开始："
)

response = client.chat.completions.create(
    model=config['llm']['model'],
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': augmented_prompt}
    ],
    tools=tools,
    tool_choice='auto',
    stream=True
)

for chunk in response:
    if not chunk.choices: continue
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        print("TOOL CALL CHUNK:", [tc.model_dump() for tc in delta.tool_calls])
    if delta.content:
        print("CONTENT CHUNK:", repr(delta.content))
