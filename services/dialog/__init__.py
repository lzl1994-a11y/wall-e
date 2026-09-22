"""Conversation turns, expression policy, output guards, and dialog workflow.

English
-------
Dialog services coordinate one user-facing interaction: turn identifiers,
sentence boundaries, TTS completion, wake/listen transitions, expression poses,
camera inspection, and final response decisions.  External operations are
provided as callbacks or narrow service interfaces so the workflow can be
tested without ROS, microphones, models, or motors.  Provider protocols belong
to speech/LLM packages and device protocols belong to hardware/display packages.

中文
----
对话层协调一次面向用户的完整交互，包括轮次标识、分句、TTS 完成、唤醒与恢复监听、
表情姿态、相机查看和最终回复决策。所有外部操作都通过回调或窄接口注入，因此无需 ROS、
麦克风、模型或电机也能测试工作流。供应商协议属于语音/大模型包，设备协议属于硬件或
显示包，对话代码只负责组合它们，不重新实现底层细节。
"""
