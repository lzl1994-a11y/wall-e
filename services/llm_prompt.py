"""Shared prompt rules for text that is sent directly to the robot speaker."""


DIRECT_SPEECH_POLICY = """
输出内容会直接送到扬声器。只给最终台词，不要输出思考、分析、计划、步骤、规则复述、
Markdown、括号说明、工具名、工具参数或工具调用过程。需要动作时使用原生工具调用，
不要在台词中描述动作执行过程。回答保持简短自然，通常不超过两句。
""".strip()


def with_direct_speech_policy(system_prompt):
    base = str(system_prompt or "").strip()
    if not base:
        return DIRECT_SPEECH_POLICY
    return f"{base}\n\n{DIRECT_SPEECH_POLICY}"
