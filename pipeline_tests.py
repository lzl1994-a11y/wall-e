#!/usr/bin/env python3
"""Runtime test launchers for the Wali ROS2 pipeline.

This file intentionally lives outside `nodes/` and `services/`. It starts the
existing ROS node scripts as child processes, then provides two test classes:

- KeyboardInputLLMTest: manual text -> voice_text -> LLM -> serial/screen path.
- MicrophoneLLMTest: stt_ros_node microphone -> voice_text -> LLM -> serial path.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NODES_DIR = ROOT / "nodes"


class NodeProcessRunner:
    """Start and stop existing node scripts as one test process group."""

    def __init__(self, python_executable=None, no_serial=False, startup_gap=0.5):
        self.python_executable = python_executable or sys.executable
        self.no_serial = no_serial
        self.startup_gap = startup_gap
        self.processes = []

    def node_script(self, filename):
        script = NODES_DIR / filename
        if not script.exists():
            raise FileNotFoundError(f"Node script not found: {script}")
        return script

    def start_node(self, name, filename):
        script = self.node_script(filename)
        cmd = [self.python_executable, str(script)]
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

        print(f"[test-launcher] starting {name}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, **kwargs)
        self.processes.append((name, proc))
        time.sleep(self.startup_gap)

    def start_llm_and_serial(self):
        self.start_node("llm", "llm_ros_node.py")
        if not self.no_serial:
            self.start_node("serial", "serial_ros_node.py")

    def assert_running(self):
        for name, proc in self.processes:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(f"Node {name} exited with code {code}")

    def wait_forever(self):
        while True:
            self.assert_running()
            time.sleep(1.0)

    def stop_all(self, timeout=5.0):
        for name, proc in reversed(self.processes):
            self.stop_process(name, proc, timeout=timeout)

    @staticmethod
    def stop_process(name, proc, timeout=5.0):
        if proc.poll() is not None:
            return

        print(f"[test-launcher] stopping {name}...")
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
            print(f"[test-launcher] killing {name}...")
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()


def read_console_text():
    sys.stdout.write("voice_text> ")
    sys.stdout.flush()

    raw = sys.stdin.buffer.readline()
    if raw == b"":
        raise EOFError

    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def safe_shutdown(rclpy):
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


class KeyboardInputLLMTest:
    """Manual text publisher for testing the LLM pipeline without STT."""

    def __init__(self, no_serial=False, python_executable=None, startup_gap=0.5):
        self.runner = NodeProcessRunner(
            python_executable=python_executable,
            no_serial=no_serial,
            startup_gap=startup_gap,
        )

    def run(self):
        self.runner.start_llm_and_serial()
        print("[keyboard-test] Type text to publish it to /voice_text.")
        print("[keyboard-test] Type /exit to stop.")
        print("[keyboard-test] Wait for 'LLM service initialized.' before the first input.")

        import rclpy
        from rclpy.executors import ExternalShutdownException
        from std_msgs.msg import String

        rclpy.init(args=None)
        node = rclpy.create_node("keyboard_input_llm_test")
        publisher = node.create_publisher(String, "voice_text", 10)

        try:
            while rclpy.ok():
                self.runner.assert_running()
                try:
                    text = read_console_text()
                except EOFError:
                    break

                if not text:
                    continue
                if text.lower() in {"/exit", "exit", "quit", "/quit"}:
                    break

                msg = String()
                msg.data = text
                publisher.publish(msg)
                node.get_logger().info(f"Published voice_text: {text}")
                rclpy.spin_once(node, timeout_sec=0.0)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
            safe_shutdown(rclpy)
            self.runner.stop_all()


class MicrophoneLLMTest:
    """Full microphone STT -> LLM -> serial/screen pipeline test."""

    def __init__(self, no_serial=False, python_executable=None, startup_gap=0.5):
        self.runner = NodeProcessRunner(
            python_executable=python_executable,
            no_serial=no_serial,
            startup_gap=startup_gap,
        )

    def run(self):
        try:
            self.runner.start_llm_and_serial()
            self.runner.start_node("stt", "stt_ros_node.py")
            print("[microphone-test] Full microphone pipeline is running.")
            print("[microphone-test] Speak into the microphone. Press Ctrl+C to stop.")
            self.runner.wait_forever()
        except KeyboardInterrupt:
            print("\n[microphone-test] Ctrl+C received.")
        finally:
            self.runner.stop_all()


def add_common_args(parser):
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Do not start serial_ros_node.py.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python executable used to start node scripts. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--startup-gap",
        type=float,
        default=0.5,
        help="Seconds to wait after starting each node script.",
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run end-to-end Wali ROS2 pipeline tests.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    keyboard = subparsers.add_parser(
        "keyboard",
        help="Start LLM/serial nodes and publish manual text to /voice_text.",
    )
    add_common_args(keyboard)

    microphone = subparsers.add_parser(
        "microphone",
        help="Start STT/LLM/serial nodes for full microphone testing.",
    )
    add_common_args(microphone)

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.mode == "keyboard":
        KeyboardInputLLMTest(
            no_serial=args.no_serial,
            python_executable=args.python_executable,
            startup_gap=args.startup_gap,
        ).run()
    elif args.mode == "microphone":
        MicrophoneLLMTest(
            no_serial=args.no_serial,
            python_executable=args.python_executable,
            startup_gap=args.startup_gap,
        ).run()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
