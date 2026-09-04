#!/usr/bin/env python3
"""Direct serial debugger for Wall-E's tracks.

This tool speaks the same ``pca9685:`` line protocol as
``hardware_bridge_node -> serial_ros_node``.  It intentionally keeps channels
0--8 at the configured servo ``init`` values, so testing a track does not move
the head, eyes, arms, or neck.

Run only when the normal serial ROS node is stopped: it is the production
owner of the ESP32 USB serial port.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import serial
import yaml

# Permit running this file directly from tools/diagnostics.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.motor_control import apply_direction_inversion, motor_inversion_flags
from services.usb_devices import serial_ports_for_role


SERVO_CHANNELS = 9
TOTAL_CHANNELS = 15
MOTOR_HIGH = 65535
HEARTBEAT_SECONDS = 0.10  # Must stay comfortably below the 300 ms watchdog.


def load_hardware_config(config_path: Path) -> dict:
    """Load the shared configuration without modifying it."""
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是映射: {config_path}")
    return config


def initial_state(config: dict) -> list[int]:
    """Build the exact 15-channel startup packet used by the hardware bridge."""
    state = [4915] * TOTAL_CHANNELS  # Same 90-degree fallback as the bridge.
    servos = config.get("servos", [])
    if not isinstance(servos, list):
        raise ValueError("config.servos 必须是列表")
    for servo_config in servos:
        if not isinstance(servo_config, dict):
            continue
        channel = servo_config.get("id")
        initial = servo_config.get("init")
        if isinstance(channel, int) and 0 <= channel < SERVO_CHANNELS and initial is not None:
            state[channel] = int(initial)
    # ch9--14: left IN1/IN2/PWM, right IN1/IN2/PWM.  Always start stopped.
    state[SERVO_CHANNELS:] = [0] * (TOTAL_CHANNELS - SERVO_CHANNELS)
    return state


def resolve_port(config: dict, explicit_port: str | None, config_path: Path) -> str:
    if explicit_port:
        return explicit_port
    selected_ports, configured = serial_ports_for_role("screen_motion", config_path)
    if configured:
        if len(selected_ports) == 1:
            return selected_ports[0]
        if not selected_ports:
            raise RuntimeError("已配置 screen_motion USB，但当前未发现对应串口")
        raise RuntimeError(f"screen_motion USB 匹配到多个串口，请用 --port 指定: {selected_ports}")
    port = config.get("serial", {}).get("lower_board_port")
    if not isinstance(port, str) or not port.strip() or port == "COM_ESP32S3":
        raise RuntimeError("未配置下位机串口，请用 --port COMx 或填写 serial.lower_board_port")
    return port.strip()


def encode_packet(state: Iterable[int]) -> bytes:
    values = [int(value) for value in state]
    if len(values) != TOTAL_CHANNELS:
        raise ValueError(f"PCA9685 数据必须有 {TOTAL_CHANNELS} 个通道")
    return ("pca9685:" + ",".join(map(str, values)) + "\n").encode("ascii")


def set_motor(state: list[int], base_channel: int, action: int, speed: int) -> None:
    """Apply one track using the exact IN1/IN2/PWM mapping of the bridge."""
    if action == 1:
        in1, in2 = MOTOR_HIGH, 0
    elif action == 2:
        in1, in2 = 0, MOTOR_HIGH
    else:
        in1 = in2 = speed = 0
    state[base_channel : base_channel + 3] = [in1, in2, int(speed / 100 * MOTOR_HIGH)]


def motion_state(config: dict, motion: str, speed: int) -> list[int]:
    state = initial_state(config)
    logical_actions = {
        "forward": (1, 1),
        "backward": (2, 2),
        "left": (2, 1),
        "right": (1, 2),
        "stop": (0, 0),
    }[motion]
    inverted = motor_inversion_flags(config.get("motors"))
    left_action = apply_direction_inversion(logical_actions[0], inverted["left"])
    right_action = apply_direction_inversion(logical_actions[1], inverted["right"])
    set_motor(state, 9, left_action, speed)
    set_motor(state, 12, right_action, speed)
    return state


def send_for_duration(stream, packet: bytes, duration: float, stop_event=None) -> None:
    """Send a regular heartbeat for the requested motion time."""
    deadline = time.monotonic() + duration
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        stream.write(packet)
        stream.flush()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        pause = min(HEARTBEAT_SECONDS, remaining)
        if stop_event is None:
            time.sleep(pause)
        else:
            # The GUI emergency-stop button wakes this wait immediately.
            stop_event.wait(pause)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wall-E 履带串口调试（仅修改电机通道）")
    parser.add_argument("motion", choices=("forward", "backward", "left", "right", "stop"))
    parser.add_argument("--speed", type=int, default=20, help="速度百分比，1-100（默认 20）")
    parser.add_argument("--duration", type=float, default=0.5, help="动作秒数，默认 0.5")
    parser.add_argument("--port", help="直接指定下位机串口，例如 COM8 或 /dev/ttyACM0")
    parser.add_argument("--config", type=Path, default=ROOT / "core" / "config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只打印数据包，不打开串口")
    args = parser.parse_args()
    if not 1 <= args.speed <= 100:
        parser.error("--speed 必须在 1 到 100 之间")
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    try:
        config = load_hardware_config(args.config)
        moving = encode_packet(motion_state(config, args.motion, args.speed))
        stopped = encode_packet(motion_state(config, "stop", 0))
        if args.dry_run:
            print(moving.decode().strip())
            print(stopped.decode().strip())
            return 0
        port = resolve_port(config, args.port, args.config)
        print(f"连接 {port}；{args.motion}，速度 {args.speed}%，{args.duration:g} 秒")
        # Do not set DTR/RTS: toggling them can reboot the ESP32 while testing.
        with serial.Serial(port, baudrate=115200, timeout=1, write_timeout=2) as stream:
            try:
                send_for_duration(stream, moving, args.duration)
            finally:
                # A final stop packet is mandatory even on Ctrl+C or write errors.
                stream.write(stopped)
                stream.flush()
        print("履带已停车；舵机仍保持 config.yaml 的 init 值。")
        return 0
    except (OSError, serial.SerialException, RuntimeError, ValueError) as exc:
        print(f"调试失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
