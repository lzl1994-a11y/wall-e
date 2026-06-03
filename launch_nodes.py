#!/usr/bin/env python3
"""Start the local ROS2 Python nodes for Wali.

Default test pipeline:
  keyboard_stt_node.py -> voice_text -> llm_ros_node.py -> screen_dialog -> serial_ros_node.py

Examples:
  python3 launch_nodes.py
  python3 launch_nodes.py --real-stt
  python3 launch_nodes.py --no-serial
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# 默认自动重连：最多 5 次，每次间隔 3 秒
DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_DELAY = 3.0


@dataclass
class NodeEntry:
    """launcher 内部对每个受管节点的元信息"""
    name: str
    script: Path
    max_restarts: int = DEFAULT_MAX_RESTARTS
    restart_delay: float = DEFAULT_RESTART_DELAY


def build_node_list(args):
    nodes = [NodeEntry("llm", ROOT / "nodes" / "llm_ros_node.py")]

    if not args.no_serial:
        nodes.append(NodeEntry("serial", ROOT / "nodes" / "serial_ros_node.py"))

    if args.real_stt:
        nodes.append(NodeEntry("stt", ROOT / "nodes" / "stt_ros_node.py"))
    else:
        nodes.append(NodeEntry("keyboard_stt", ROOT / "nodes" / "keyboard_stt_node.py"))

    # ── 视觉跟踪相关节点（通过 --tracking 启用）──
    if args.tracking:
        nodes.append(NodeEntry("tracking", ROOT / "nodes" / "wali_tracking_node.py"))
        if not args.no_hardware:
            nodes.append(NodeEntry("servo_ros", ROOT / "nodes" / "servo_ros_node.py"))
            nodes.append(NodeEntry("motor_ros", ROOT / "nodes" / "motor_ros_node.py"))
        if not args.no_doa:
            nodes.append(NodeEntry("doa_ros", ROOT / "nodes" / "doa_ros_node.py"))

    return nodes


@dataclass
class ManagedProcess:
    """受管进程：记录当前运行的 Popen 对象与重试计数"""
    entry: NodeEntry
    proc: subprocess.Popen
    restarts: int = 0


def start_process(entry: NodeEntry):
    """启动一个节点子进程，返回 ManagedProcess"""
    script = entry.script
    if not script.exists():
        raise FileNotFoundError(f"Node script not found: {script}")

    cmd = [sys.executable, str(script)]
    kwargs = {
        "cwd": str(ROOT),
        "stdin": None,
        "stdout": None,
        "stderr": None,
    }

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    print(f"[launcher] starting {entry.name}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, **kwargs)
    return ManagedProcess(entry=entry, proc=proc, restarts=0)


def stop_managed(mp: ManagedProcess, timeout=5.0):
    """关闭一个受管进程（发送 SIGINT / Ctrl+C）"""
    proc = mp.proc
    if proc.poll() is not None:
        return

    print(f"[launcher] stopping {mp.entry.name}...")
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[launcher] killing {mp.entry.name}...")
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def restart_managed(mp: ManagedProcess):
    """重启一个受管进程（先停旧进程再起新的），返回新的 ManagedProcess"""
    stop_managed(mp, timeout=2.0)
    time.sleep(0.5)
    new_mp = start_process(mp.entry)
    new_mp.restarts = mp.restarts + 1
    return new_mp


def main():
    parser = argparse.ArgumentParser(description="Start Wali ROS2 Python nodes.")
    parser.add_argument(
        "--real-stt",
        action="store_true",
        help="Start stt_ros_node.py instead of keyboard_stt_node.py.",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Do not start serial_ros_node.py.",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Start visual tracking nodes (servo_ros, motor_ros, wali_tracking, doa_ros).",
    )
    parser.add_argument(
        "--no-doa",
        action="store_true",
        help="When --tracking is active, skip doa_ros_node.",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="When --tracking is active, skip servo_ros and motor_ros (no PCA9685).",
    )
    args = parser.parse_args()

    entries = build_node_list(args)
    managed = []                     # 运行中的受管进程列表
    stopped = False                  # Ctrl+C 标志位，避免重启循环

    # 设置 Ctrl+C 处理器，只设置标志位
    def _sigint_handler(sig, frame):
        nonlocal stopped
        if stopped:
            return
        stopped = True
        print("\n[launcher] Ctrl+C received, shutting down...")

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # 首次启动全部节点
        for entry in entries:
            managed.append(start_process(entry))
            time.sleep(0.5)

        print("[launcher] all nodes started.")
        if not args.real_stt:
            print("[launcher] type text at the 'voice_text>' prompt to simulate STT.")
        print("[launcher] press Ctrl+C to stop all nodes.")

        # 主监控循环：节点崩溃 → 自动重启（有上限）
        while not stopped:
            for i, mp in enumerate(managed):
                code = mp.proc.poll()
                if code is not None:
                    # 某个节点退出了
                    name = mp.entry.name
                    max_r = mp.entry.max_restarts

                    if mp.restarts >= max_r:
                        print(f"[launcher] Node {name} exited with code {code} "
                              f"(restarts={mp.restarts}/{max_r}), stopping permanently.")
                    else:
                        print(f"[launcher] Node {name} exited with code {code}, "
                              f"restarting in {mp.entry.restart_delay:.0f}s "
                              f"(attempt {mp.restarts + 1}/{max_r})...")
                        time.sleep(mp.entry.restart_delay)
                        if stopped:
                            break
                        # 重启并替换列表中的旧条目
                        managed[i] = restart_managed(mp)

            time.sleep(1.0)

    except Exception as exc:
        if not stopped:
            print(f"[launcher] error: {exc}")
    finally:
        for mp in reversed(managed):
            stop_managed(mp)
        print("[launcher] shutdown complete.")


if __name__ == "__main__":
    main()
