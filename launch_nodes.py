#!/usr/bin/env python3
"""Start the local ROS2 Python nodes for Wali.

Voice pipeline is set in core/config.yaml → launch.voice_pipeline
and can be overridden by CLI flags: --voice-chat / --real-stt / --keyboard-stt.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "core" / "config.yaml"

# 默认自动重连：最多 5 次，每次间隔 3 秒
DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_DELAY = 3.0


def load_config():
    if yaml is None:
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@dataclass
class NodeEntry:
    """launcher 内部对每个受管节点的元信息"""
    name: str
    script: Path
    max_restarts: int = DEFAULT_MAX_RESTARTS
    restart_delay: float = DEFAULT_RESTART_DELAY


def build_node_list(args):
    config = load_config()
    launch_cfg = config.get("launch", {})

    # voice_pipeline: CLI 优先 → config → keyboard
    if args.voice_chat:
        pipeline = "voice_chat"
    elif args.real_stt:
        pipeline = "real_stt"
    elif args.keyboard_stt:
        pipeline = "keyboard"
    else:
        pipeline = launch_cfg.get("voice_pipeline", "keyboard")

    nodes = [NodeEntry("llm", ROOT / "nodes" / "llm_ros_node.py")]

    # serial: CLI --no-serial 覆盖 config
    if not args.no_serial and launch_cfg.get("serial", True):
        nodes.append(NodeEntry("serial", ROOT / "nodes" / "serial_ros_node.py"))

    if pipeline == "voice_chat":
        nodes.append(NodeEntry("voice_chat", ROOT / "nodes" / "voice_chat_ros_node.py"))
        nodes = [n for n in nodes if n.name != "llm"]
    elif pipeline == "real_stt":
        nodes.append(NodeEntry("stt", ROOT / "nodes" / "stt_ros_node.py"))
    else:
        nodes.append(NodeEntry("keyboard_stt", ROOT / "nodes" / "keyboard_stt_node.py"))

    # tracking: CLI --tracking 覆盖 config
    if args.tracking or launch_cfg.get("tracking", False):
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
        "--voice-chat",
        action="store_true",
        help="Use Qwen-Omni audio→LLM pipeline (replaces stt+llm).",
    )
    parser.add_argument(
        "--real-stt",
        action="store_true",
        help="Use stt_ros_node.py (Aliyun Paraformer).",
    )
    parser.add_argument(
        "--keyboard-stt",
        action="store_true",
        help="Use keyboard_stt_node.py (text simulation).",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Do not start serial_ros_node.py.",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Start visual tracking nodes.",
    )
    parser.add_argument(
        "--no-doa",
        action="store_true",
        help="When tracking is active, skip doa_ros_node.",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="When tracking is active, skip servo_ros and motor_ros.",
    )
    args = parser.parse_args()

    entries = build_node_list(args)
    managed = []
    stopped = False

    def _sigint_handler(sig, frame):
        nonlocal stopped
        if stopped:
            return
        stopped = True
        print("\n[launcher] Ctrl+C received, shutting down...")

    signal.signal(signal.SIGINT, _sigint_handler)

    # print which pipeline is active
    names = [e.name for e in entries]
    print(f"[launcher] nodes: {', '.join(names)}")

    try:
        for entry in entries:
            managed.append(start_process(entry))
            time.sleep(0.5)

        print("[launcher] all nodes started. press Ctrl+C to stop.")

        while not stopped:
            for i, mp in enumerate(managed):
                code = mp.proc.poll()
                if code is not None:
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