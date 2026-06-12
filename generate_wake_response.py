#!/usr/bin/env python3
"""预生成唤醒回复语音文件。

使用 edge-tts 合成"我在，今天你想聊什么"，输出 WAV 到 assets/wake_response.wav。
机器人启动前运行一次即可。
"""

import asyncio
import os
import wave
from pathlib import Path

import edge_tts
import pydub  # pip install pydub

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "wake_response.wav"

VOICE = "zh-CN-XiaoxiaoNeural"
TEXT = "我在，今天你想聊什么"


async def synthesize():
    os.makedirs(OUTPUT.parent, exist_ok=True)

    # edge-tts → MP3 bytes
    print(f"合成: '{TEXT}' (voice={VOICE})")
    communicate = edge_tts.Communicate(TEXT, VOICE, rate="+10%", pitch="+0Hz")
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk["data"]

    if not mp3_data:
        raise RuntimeError("edge-tts 返回空数据，请检查网络或 voice 名称")

    # MP3 → WAV (pydub)
    audio = pydub.AudioSegment.from_mp3(pydub.utils.mediainfo_bytes(mp3_data))
    # 转 16kHz 16-bit mono，与机器人音频管线一致
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    audio.export(OUTPUT, format="wav")
    print(f"已保存: {OUTPUT}")
    print(f"时长: {len(audio) / 1000:.1f}s, "
          f"采样率: {audio.frame_rate}Hz, "
          f"声道: {audio.channels}")


if __name__ == "__main__":
    asyncio.run(synthesize())
