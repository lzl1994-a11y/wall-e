"""
视觉子进程：摄像头读取、YOLO 推理、PID 误差计算。
核心机制：受控于跨进程共享的 vision_enabled 闸门变量。
"""

from time import sleep


def run_vision_service(bus) -> None:
    while True:
        if not bus.vision_enabled.value:
            sleep(0.05)
            continue

        # TODO: 接入摄像头读取、YOLO 推理和 PID 误差计算。
        sleep(0.02)
