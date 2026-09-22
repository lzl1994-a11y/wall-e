"""PCM capture, buffering, mixing, playback, and music processing.

音频层集中处理采集、缓冲、混音、播放、节拍控制和频谱数据。它只维护音频数据与
播放状态，不决定对话内容或业务动作，避免音频设备逻辑渗入业务层。
"""
