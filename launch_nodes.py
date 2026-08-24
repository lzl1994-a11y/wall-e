#!/usr/bin/env python3
"""Start the local ROS2 Python nodes for Wali.

Voice pipeline is set in core/config.yaml → pipeline.mode
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
    environment_setup: Path | None = None


def build_node_list(args):
    config = load_config()
    launch_cfg = config.get("launch", {})
    hardware_cfg = config.get("hardware", {})
    mcp_cfg = config.get("mcp", {})
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}
    if not isinstance(hardware_cfg, dict):
        hardware_cfg = {}
    hardware_backend = hardware_cfg.get("backend", "serial_mcu")
    if hardware_backend not in {"serial_mcu", "ubuntu_i2c"}:
        hardware_backend = "serial_mcu"

    # pipeline mode: CLI 优先 → config → keyboard
    if args.voice_chat:
        pipeline = "multimodal"
    elif args.real_stt:
        pipeline = "asr_llm"
    elif args.keyboard_stt:
        pipeline = "keyboard"
    else:
        pipeline = config.get("pipeline", {}).get("mode", "keyboard")

    nodes = [NodeEntry("camera_capture", ROOT / "nodes" / "camera_capture_node.py")]
    if not args.no_web:
        nodes.append(NodeEntry("config_web", ROOT / "services" / "web_server.py"))

    nodes.append(NodeEntry("llm", ROOT / "nodes" / "llm_ros_node.py"))

    # 音频播放管线（始终启动）
    nodes.append(NodeEntry("tts_play", ROOT / "nodes" / "tts_play_node.py"))
    nodes.append(NodeEntry("audio_playback", ROOT / "nodes" / "audio_playback_node.py"))

    # Screen/motion control cluster. The selected hardware backend has one owner.
    if launch_cfg.get("serial", True):
        if not args.no_serial:
            nodes.append(NodeEntry("serial", ROOT / "nodes" / "serial_ros_node.py"))
        nodes.append(NodeEntry("motion_arbiter", ROOT / "nodes" / "motion_arbiter_node.py"))
        nodes.append(NodeEntry("action", ROOT / "nodes" / "sequence_ros_node.py"))
        mcp_enabled = (
            getattr(args, "mcp", False)
            or (
                mcp_cfg.get("enabled", False)
                and not getattr(args, "no_mcp", False)
            )
        )
        if mcp_enabled:
            nodes.append(NodeEntry("mcp_gateway", ROOT / "nodes" / "wali_mcp_server.py"))
        nodes.append(NodeEntry("joy_control", ROOT / "nodes" / "joy_control_node.py"))
        if not args.no_hardware:
            if hardware_backend == "ubuntu_i2c":
                nodes.append(NodeEntry("i2c_hardware", ROOT / "nodes" / "i2c_hardware_node.py"))
            elif not args.no_serial:
                nodes.append(NodeEntry("hardware_bridge", ROOT / "nodes" / "hardware_bridge_node.py"))

    if pipeline == "multimodal":
        nodes.append(NodeEntry("voice_chat", ROOT / "nodes" / "voice_chat_ros_node.py"))
        nodes = [n for n in nodes if n.name != "llm"]
    elif pipeline == "asr_llm":
        nodes.append(NodeEntry("stt", ROOT / "nodes" / "stt_ros_node.py"))
    else:
        nodes.append(NodeEntry("keyboard_stt", ROOT / "nodes" / "keyboard_stt_node.py"))

    # tracking: CLI --tracking 覆盖 config
    if args.tracking or launch_cfg.get("tracking", False):
        nodes.append(NodeEntry("hobot_vision", ROOT / "nodes" / "hobot_vision_node.py"))
        nodes.append(
            NodeEntry(
                "tracking",
                ROOT / "nodes" / "wali_tracking_node.py",
                environment_setup=Path("/opt/tros/humble/setup.bash"),
            )
        )
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
    if entry.environment_setup is not None and os.name != "nt":
        # Positional parameters keep paths safely quoted while sourcing the RDK
        # environment in the same shell that execs the ROS node.
        cmd = [
            "bash",
            "-c",
            'source "$1" && exec "$2" "$3"',
            "wali-node",
            str(entry.environment_setup),
            sys.executable,
            str(script),
        ]
    env = os.environ.copy()
    root_path = str(ROOT)
    existing_paths = [
        path
        for path in env.get("PYTHONPATH", "").split(os.pathsep)
        if path and path != root_path
    ]
    env["PYTHONPATH"] = os.pathsep.join([root_path, *existing_paths])
    kwargs = {
        "cwd": str(ROOT),
        "env": env,
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
        help="Do not start the configured servo/motor hardware backend.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Do not start the config web service.",
    )
    mcp_group = parser.add_mutually_exclusive_group()
    mcp_group.add_argument(
        "--mcp",
        action="store_true",
        help="Start the authenticated Streamable HTTP MCP gateway.",
    )
    mcp_group.add_argument(
        "--no-mcp",
        action="store_true",
        help="Do not start MCP even when mcp.enabled is true.",
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
