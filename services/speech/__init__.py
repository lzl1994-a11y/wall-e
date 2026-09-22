"""Speech recognition, synthesis, provider adapters, and voice diagnostics.

English
-------
Speech services expose stable audio-to-text and text-to-audio operations while
hiding cloud/local provider differences.  The ASR factory validates model paths
and selects an adapter; STT/TTS services coordinate those adapters with the
shared audio pipeline; rolling diagnostics retain bounded artifacts only when
explicitly enabled.  This package does not decide what the robot should say or
which physical action it should execute.

中文
----
语音层提供稳定的“音频转文字”和“文字转音频”接口，并隐藏云端与本地供应商差异。ASR
工厂负责校验模型路径和选择适配器，STT/TTS 服务把适配器接入公共音频管线，滚动调试
存储只在显式开启时保留有限数量的诊断文件。本层不决定瓦力应该说什么，也不决定应该
执行哪个物理动作，这些业务判断属于对话、大模型和动作层。
"""
