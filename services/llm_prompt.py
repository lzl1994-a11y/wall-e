"""Shared prompt rules for text that is sent directly to the robot speaker."""


DIRECT_SPEECH_POLICY = """
输出内容会直接送到扬声器。只给最终台词，不要输出思考、分析、计划、步骤、规则复述、
Markdown、括号说明、工具名、工具参数或工具调用过程。需要动作时使用原生工具调用，
不要在台词中描述动作执行过程。普通对话保持简短自然，通常不超过两句；
朗读、背诵或完整内容请求按用户要求连续完整输出。
""".strip()


STRUCTURED_ANSWER_POLICY = """
本轮必须调用 direct_answer 工具一次，把唯一可播报的最终台词写入 response 参数。
即使还需要调用身体动作工具，也必须同时调用 direct_answer。不要在普通 content 中输出
台词、思考或解释；普通 content 不会被播放。
""".strip()


def with_direct_speech_policy(system_prompt):
    base = str(system_prompt or "").strip()
    if not base:
        return DIRECT_SPEECH_POLICY
    return f"{base}\n\n{DIRECT_SPEECH_POLICY}"


def with_structured_answer_policy(system_prompt):
    """Require a native tool-call answer for a tools-enabled voice turn."""
    base = str(system_prompt or "").strip()
    return f"{base}\n\n{STRUCTURED_ANSWER_POLICY}" if base else STRUCTURED_ANSWER_POLICY
