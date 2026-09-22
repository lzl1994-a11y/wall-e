"""Speech recognition, synthesis, and voice diagnostics boundaries.

语音层封装 ASR、STT、TTS 管线及语音调试能力，把音频与文本之间的转换统一为稳定
接口；模型供应商和实现差异保留在适配器内部，不向对话流程扩散。
"""
