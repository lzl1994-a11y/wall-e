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
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_node_list(args):
    nodes = [("llm", ROOT / "nodes" / "llm_ros_node.py")]

    if not args.no_serial:
        nodes.append(("serial", ROOT / "nodes" / "serial_ros_node.py"))

    if args.real_stt:
        nodes.append(("stt", ROOT / "nodes" / "stt_ros_node.py"))
    else:
        nodes.append(("keyboard_stt", ROOT / "nodes" / "keyboard_stt_node.py"))

    return nodes


def start_process(name, script):
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

    print(f"[launcher] starting {name}: {' '.join(cmd)}")
    return name, subprocess.Popen(cmd, **kwargs)


def stop_process(name, proc, timeout=5.0):
    if proc.poll() is not None:
        return

    print(f"[launcher] stopping {name}...")
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
        print(f"[launcher] killing {name}...")
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


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
    args = parser.parse_args()

    processes = []
    try:
        for name, script in build_node_list(args):
            processes.append(start_process(name, script))
            time.sleep(0.5)

        print("[launcher] all nodes started.")
        if not args.real_stt:
            print("[launcher] type text at the 'voice_text>' prompt to simulate STT.")
        print("[launcher] press Ctrl+C to stop all nodes.")

        while True:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(f"Node {name} exited with code {code}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received.")
    except Exception as exc:
        print(f"[launcher] error: {exc}")
    finally:
        for name, proc in reversed(processes):
            stop_process(name, proc)
        print("[launcher] shutdown complete.")


if __name__ == "__main__":
    main()
