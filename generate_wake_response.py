#!/usr/bin/env python3
"""预生成不会在运行期调用云服务的短提示音文件。

使用 edge-tts 合成唤醒和 ESP32 配网提示，输出 48 kHz 单声道 PCM WAV。
机器人启动前运行一次即可。
"""

import asyncio
import argparse
import os
import wave
from pathlib import Path

import edge_tts
import pydub  # pip install pydub

from services.audio_output import (
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_WIDTH,
)

ROOT = Path(__file__).resolve().parent
VOICE = "zh-CN-YunxiaNeural"
PROMPTS = {
    "wake_response.wav": "我在，今天你想聊什么",
    "esp_network_configuring.wav": "ESP配网中",
    "esp_network_connected.wav": "ESP配网成功",
}


async def synthesize(names=None):
    output_dir = ROOT / "assets"
    os.makedirs(output_dir, exist_ok=True)
    selected = PROMPTS if not names else {name: PROMPTS[name] for name in names}
    for filename, text in selected.items():
        output = output_dir / filename
        print(f"合成: '{text}' (voice={VOICE})")
        communicate = edge_tts.Communicate(text, VOICE, rate="+20%", pitch="+5Hz")
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk["data"]
        if not mp3_data:
            raise RuntimeError("edge-tts 返回空数据，请检查网络或 voice 名称")

        import io
        audio = pydub.AudioSegment.from_mp3(io.BytesIO(mp3_data))
        audio = (
            audio.set_frame_rate(OUTPUT_SAMPLE_RATE)
            .set_channels(OUTPUT_CHANNELS)
            .set_sample_width(OUTPUT_SAMPLE_WIDTH)
        )
        audio.export(output, format="wav")
        print(f"已保存: {output}")
        print(f"时长: {len(audio) / 1000:.1f}s, "
              f"采样率: {audio.frame_rate}Hz, "
              f"声道: {audio.channels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=tuple(PROMPTS))
    args = parser.parse_args()
    asyncio.run(synthesize(args.names))
