#!/usr/bin/env python3
"""
地平线视觉推理进程的 Python 包装器
通过 Python 脚本拉起跟踪检测进程组。
物理摄像头由 camera_capture_node 独占；本节点只消费其 /image 帧源。
这样可以完美融入现有的 launch_nodes.py 进程管理池中，并在 Ctrl+C 时一并被干净地回收。
"""

import subprocess
import sys
import os
import signal
import time
import shlex
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.vision_pipeline_protocol import (
    VISION_PIPELINE_COMMAND_TOPIC,
    VISION_PIPELINE_START,
    decode_vision_pipeline_command,
)


class VisionPipelineControl(Node):
    """Keep detector consumers alive while allowing tracking to stop."""

    def __init__(self):
        super().__init__("hobot_vision_control")
        # ``launch.tracking`` only loads this controller so semantic voice
        # commands can use it later.  The expensive camera/detector pipeline
        # must remain off until wali_tracking_node publishes an explicit
        # VISION_PIPELINE_START command.
        self.enabled = False
        self.decoded_frames = 0
        self.padded_frames = 0
        command_qos = QoSProfile(depth=1)
        command_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            VISION_PIPELINE_COMMAND_TOPIC,
            self._on_command,
            command_qos,
        )
        self.create_subscription(
            Image,
            "/image_nv12",
            self._on_decoded_frame,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/image_padded_nv12",
            self._on_padded_frame,
            qos_profile_sensor_data,
        )

    def _on_command(self, message):
        command = decode_vision_pipeline_command(message.data)
        if command is not None:
            if command == VISION_PIPELINE_START and not self.enabled:
                self.decoded_frames = 0
                self.padded_frames = 0
            self.enabled = command == VISION_PIPELINE_START

    def _on_decoded_frame(self, _message):
        if self.enabled:
            self.decoded_frames += 1

    def _on_padded_frame(self, _message):
        if self.enabled:
            self.padded_frames += 1


def cleanup_old_processes():
    process_names = [
        "hobot_codec_republish",
        "mono2d_body_detection",
        "nv12_padder_node",
    ]
    for name in process_names:
        subprocess.run(
            ["killall", "-9", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    stale_python_nodes = [
        "nv12_padder_node.py",
        "ai_msg_sync_node.py",
        "ai_msg_scaler_node.py",
    ]
    for pattern in stale_python_nodes:
        subprocess.run(
            ["pkill", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _ensure_padder_binary(root_dir: Path, env: dict[str, str]) -> Path:
    """Build the local padder when its C++ source is newer than the binary."""
    configured = str(env.get("WALI_NV12_PADDER", "")).strip()
    if configured:
        binary = Path(configured)
        if not binary.exists():
            raise RuntimeError(f"configured NV12 padder not found: {binary}")
        return binary

    binary = root_dir / "build" / "wali_nv12_padder" / "nv12_padder_node"
    source_dir = root_dir / "cpp_nodes" / "wali_nv12_padder"
    source_files = [
        source_dir / "CMakeLists.txt",
        source_dir / "src" / "nv12_padder_node.cpp",
    ]
    needs_build = not binary.exists()
    if not needs_build:
        try:
            binary_time = binary.stat().st_mtime_ns
            needs_build = any(
                source.exists() and source.stat().st_mtime_ns > binary_time
                for source in source_files
            )
        except OSError:
            needs_build = True

    if needs_build:
        build_script = root_dir / "tools" / "build_nv12_padder.sh"
        print("[hobot_vision_node] NV12 padder is missing or stale; rebuilding...")
        result = subprocess.run(["bash", str(build_script)], env=env)
        if result.returncode != 0 or not binary.exists():
            raise RuntimeError(
                f"failed to build NV12 padder (exit={result.returncode})"
            )
    return binary


def _start_pipeline(padder_bin: Path | None = None):
    env = os.environ.copy()
    node_dir = Path(__file__).resolve().parent
    root_dir = node_dir.parent
    if padder_bin is None:
        try:
            padder_bin = _ensure_padder_binary(root_dir, env)
        except RuntimeError as exc:
            print(f"[hobot_vision_node] Error: {exc}")
            return None
    print("[hobot_vision_node] Cleaning up any zombie vision processes...")
    cleanup_old_processes()
    time.sleep(1)
    
    pipeline_script = (
        "source /opt/tros/humble/setup.bash && { "
        "ros2 run hobot_codec hobot_codec_republish --ros-args -r __node:=codec_decode --log-level WARN -p channel:=1 -p in_mode:=ros -p in_format:=jpeg -p out_mode:=ros -p out_format:=nv12 -p sub_topic:=/image -p pub_topic:=/image_nv12 & "
        f"{shlex.quote(str(padder_bin))} --ros-args --log-level WARN -p input_topic:=/image_nv12 -p output_topic:=/image_padded_nv12 -p target_width:=960 -p target_height:=544 -p flip_vertical:=true -p flip_horizontal:=true & "
        # The padded image already matches the model's 960x544 coordinate
        # space. Publish final detections directly instead of passing every AI
        # message through a no-op Python scaler and another DDS boundary.
        "(cd /opt/tros/humble/lib/mono2d_body_detection && ros2 run mono2d_body_detection mono2d_body_detection --ros-args --log-level WARN -p is_shared_mem_sub:=0 -p ros_img_topic_name:=/image_padded_nv12 -p ai_msg_pub_topic_name:=/hobot_mono2d_body_detection) & "
        # Treat every stage as critical. If codec, padder, detector, or scaler
        # exits, let the wrapper reap the remaining group and restart a clean
        # pipeline instead of leaving a video-only zombie chain alive.
        "wait -n; }"
    )
    cmd = ["bash", "-c", pipeline_script]
    
    print("[hobot_vision_node] Starting detector pipeline from /image...")
    try:
        return subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    except FileNotFoundError:
        print("[hobot_vision_node] Error: 'bash' command not found.")
        return None


def _stop_pipeline(proc):
    if not proc:
        return
    if proc.poll() is not None:
        cleanup_old_processes()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5.0)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    finally:
        # Some ``ros2 run`` wrappers create descendants that can outlive the
        # parent process group.  Reap those known vision-only processes so a
        # stopped voice tracking session cannot keep detector processes alive.
        cleanup_old_processes()


def main():
    stopping = False
    proc = None
    root_dir = Path(__file__).resolve().parent.parent
    try:
        # Build before subscribing to the transient tracking command. If a
        # command arrives during compilation, DDS retains it and delivers it
        # as soon as this control node comes online.
        padder_bin = _ensure_padder_binary(root_dir, os.environ.copy())
    except RuntimeError as exc:
        print(f"[hobot_vision_node] Error: {exc}")
        return
    rclpy.init()
    control = VisionPipelineControl()
    last_health_log = time.monotonic()
    previous_health_counts = (0, 0)

    def handler(signum, frame):
        nonlocal stopping
        stopping = True
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    cleanup_old_processes()
    try:
        while not stopping:
            rclpy.spin_once(control, timeout_sec=0.2)

            now = time.monotonic()
            if control.enabled and now - last_health_log >= 5.0:
                counts = (control.decoded_frames, control.padded_frames)
                decoded_delta = counts[0] - previous_health_counts[0]
                padded_delta = counts[1] - previous_health_counts[1]
                print(
                    "[hobot_vision_node] Frame health (last 5s): "
                    f"decoded={decoded_delta} padded={padded_delta}"
                )
                previous_health_counts = counts
                last_health_log = now

            if not control.enabled:
                if proc:
                    print("[hobot_vision_node] Tracking disabled; stopping vision pipeline")
                    _stop_pipeline(proc)
                    proc = None
                previous_health_counts = (0, 0)
                last_health_log = now
                continue

            if proc and proc.poll() is not None:
                print(
                    "[hobot_vision_node] Detector pipeline exited "
                    f"with code {proc.returncode}; restarting..."
                )
                cleanup_old_processes()
                proc = None

            if proc is None:
                proc = _start_pipeline(padder_bin)
    finally:
        print("\n[hobot_vision_node] Stopping vision pipeline...")
        _stop_pipeline(proc)
        control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
