"""PCM capture, buffering, mixing, playback, and music signal processing.

English
-------
Audio services own sample-rate conversion, buffering, silence/VAD helpers,
paced output, speech/music mixing, playback lifecycle, music decoding, and
spectrum generation.  Device discovery is delegated to ``services.hardware``;
speech-to-text and text-to-speech semantics live in ``services.speech``.  This
separation lets the mixer and output path remain reusable regardless of which
ASR, TTS, or dialog provider is selected.

中文
----
音频层负责采样率转换、缓冲、静音/VAD 辅助、节拍输出、语音与音乐混音、播放生命周期、
音乐解码和频谱生成。声卡设备发现委托给 ``services.hardware``，语音转文字和文字转语音
语义则属于 ``services.speech``。因此无论更换哪一种 ASR、TTS 或对话供应商，混音与播放
链路都能保持独立复用。
"""
