#!/usr/bin/env python3
"""
地平线视觉推理进程的 Python 包装器
通过 python 脚本拉起 ros2 launch dnn_node_example，并自动注入 CAM_TYPE=usb。
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
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.usb_devices import resolve_camera_device
from services.vision_pipeline_protocol import (
    VISION_PIPELINE_COMMAND_TOPIC,
    VISION_PIPELINE_START,
    decode_vision_pipeline_command,
)


class VisionPipelineControl(Node):
    """Keep the detector wrapper alive while allowing its camera to stop."""

    def __init__(self):
        super().__init__("hobot_vision_control")
        # ``launch.tracking`` only loads this controller so semantic voice
        # commands can use it later.  The expensive camera/detector pipeline
        # must remain off until wali_tracking_node publishes an explicit
        # VISION_PIPELINE_START command.
        self.enabled = False
        command_qos = QoSProfile(depth=1)
        command_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            VISION_PIPELINE_COMMAND_TOPIC,
            self._on_command,
            command_qos,
        )

    def _on_command(self, message):
        command = decode_vision_pipeline_command(message.data)
        if command is not None:
            self.enabled = command == VISION_PIPELINE_START


def cleanup_old_processes():
    process_names = [
        "hobot_usb_cam",
        "hobot_codec_republish",
        "mono2d_body_detection",
        "nv12_padder_node",
        "websocket",
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

def _start_pipeline(video_device):
    env = os.environ.copy()
    env["CAM_TYPE"] = "usb"
    node_dir = Path(__file__).resolve().parent
    root_dir = node_dir.parent
    ros_python = os.environ.get("WALI_ROS_PYTHON", "/usr/bin/python_backup")
    padder_bin = Path(
        os.environ.get(
            "WALI_NV12_PADDER",
            str(root_dir / "build" / "wali_nv12_padder" / "nv12_padder_node"),
        )
    )
    scaler_script = shlex.quote(str(node_dir / "ai_msg_scaler_node.py"))
    if not padder_bin.exists():
        print(
            "[hobot_vision_node] Error: fast NV12 padder binary not found: "
            f"{padder_bin}\n"
            "[hobot_vision_node] Build it first with: bash tools/build_nv12_padder.sh"
        )
        sys.exit(1)
    
    print("[hobot_vision_node] Cleaning up any zombie vision processes...")
    cleanup_old_processes()
    time.sleep(1)
    
    pipeline_script = (
        "source /opt/tros/humble/setup.bash && { "
        f"ros2 run hobot_usb_cam hobot_usb_cam --ros-args --log-level WARN -p video_device:={shlex.quote(video_device)} -p image_width:=640 -p image_height:=480 & "
        "ros2 run hobot_codec hobot_codec_republish --ros-args -r __node:=codec_decode --log-level WARN -p channel:=1 -p in_mode:=ros -p in_format:=jpeg -p out_mode:=ros -p out_format:=nv12 -p sub_topic:=/image -p pub_topic:=/image_nv12 & "
        f"{shlex.quote(str(padder_bin))} --ros-args --log-level WARN -p input_topic:=/image_nv12 -p output_topic:=/image_padded_nv12 -p target_width:=960 -p target_height:=544 -p flip_vertical:=true -p flip_horizontal:=true & "
        "ros2 run hobot_codec hobot_codec_republish --ros-args -r __node:=codec_encode --log-level WARN -p channel:=2 -p in_mode:=ros -p in_format:=nv12 -p out_mode:=ros -p out_format:=jpeg -p sub_topic:=/image_padded_nv12 -p pub_topic:=/image_padded_jpeg & "
        "(cd /opt/tros/humble/lib/mono2d_body_detection && ros2 run mono2d_body_detection mono2d_body_detection --ros-args --log-level WARN -p is_shared_mem_sub:=0 -p ros_img_topic_name:=/image_padded_nv12 -p ai_msg_pub_topic_name:=/hobot_mono2d_body_detection_raw) & "
        f"{shlex.quote(ros_python)} {scaler_script} --ros-args --log-level WARN -p transform_mode:=none -p image_topic:=/image_padded_jpeg -p model_width:=960.0 -p model_height:=544.0 -p x_scale:=1.0 -p y_scale:=1.0 -p x_offset:=0.0 -p y_offset:=0.0 -p clip_width:=960.0 -p clip_height:=544.0 & "
        "ros2 run websocket websocket --ros-args --log-level WARN -p image_topic:=/image_padded_jpeg -p image_type:=mjpeg -p msg_pub_topic_name:=/hobot_mono2d_body_detection & "
        "wait; }"
    )
    cmd = ["bash", "-c", pipeline_script]
    
    print(f"[hobot_vision_node] Starting vision pipeline with {video_device}...")
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
        # stopped voice tracking session cannot keep the camera/model alive.
        cleanup_old_processes()


def main():
    stopping = False
    proc = None
    active_device = None
    rclpy.init()
    control = VisionPipelineControl()

    def handler(signum, frame):
        nonlocal stopping
        stopping = True
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    cleanup_old_processes()
    try:
        while not stopping:
            rclpy.spin_once(control, timeout_sec=0.2)

            if not control.enabled:
                if proc:
                    print("[hobot_vision_node] Tracking disabled; stopping vision pipeline")
                    _stop_pipeline(proc)
                    proc = None
                    active_device = None
                continue

            video_device = resolve_camera_device()
            if not video_device:
                if proc:
                    print("[hobot_vision_node] Camera disconnected; stopping vision pipeline")
                    _stop_pipeline(proc)
                    proc = None
                    active_device = None
                continue

            if proc and proc.poll() is not None:
                cleanup_old_processes()
                proc = None
                active_device = None

            if proc and video_device != active_device:
                _stop_pipeline(proc)
                proc = None
                active_device = None

            if proc is None:
                proc = _start_pipeline(video_device)
                active_device = video_device if proc else None
    finally:
        print("\n[hobot_vision_node] Stopping vision pipeline...")
        _stop_pipeline(proc)
        control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
