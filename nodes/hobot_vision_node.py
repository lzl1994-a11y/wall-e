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


def cleanup_old_processes():
    process_names = [
        "hobot_usb_cam",
        "hobot_codec_republish",
        "mono2d_body_detection",
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

def main():
    env = os.environ.copy()
    env["CAM_TYPE"] = "usb"
    node_dir = Path(__file__).resolve().parent
    ros_python = os.environ.get("WALI_ROS_PYTHON", "/usr/bin/python_backup")
    scaler_script = shlex.quote(str(node_dir / "ai_msg_scaler_node.py"))
    
    print("[hobot_vision_node] Cleaning up any zombie vision processes...")
    cleanup_old_processes()
    time.sleep(1)
    
    cmd = [
        "bash", "-c",
        "source /opt/tros/humble/setup.bash && "
        "ros2 run hobot_usb_cam hobot_usb_cam --ros-args --log-level WARN -p video_device:=/dev/video0 -p image_width:=640 -p image_height:=480 & "
        "ros2 run hobot_codec hobot_codec_republish --ros-args -r __node:=codec_decode --log-level WARN -p channel:=1 -p in_mode:=ros -p in_format:=jpeg -p out_mode:=ros -p out_format:=nv12 -p sub_topic:=/image -p pub_topic:=/image_nv12 & "
        "(cd /opt/tros/humble/lib/mono2d_body_detection && ros2 run mono2d_body_detection mono2d_body_detection --ros-args --log-level WARN -p is_shared_mem_sub:=0 -p ros_img_topic_name:=/image_nv12 -p ai_msg_pub_topic_name:=/hobot_mono2d_body_detection_raw) & "
        f"{shlex.quote(ros_python)} {scaler_script} --ros-args --log-level WARN -p transform_mode:=none -p image_topic:=/image -p model_width:=960.0 -p model_height:=544.0 -p x_scale:=1.0 -p y_scale:=1.0 -p x_offset:=0.0 -p y_offset:=0.0 & "
        "ros2 run websocket websocket --ros-args --log-level WARN -p image_topic:=/image -p image_type:=mjpeg -p msg_pub_topic_name:=/hobot_mono2d_body_detection & "
        "wait"
    ]
    
    print("[hobot_vision_node] Starting direct vision pipeline...")
    
    try:
        # 使用 preexec_fn=os.setsid 将进程组分离，这样可以通过 killpg 一键杀掉全部后台 & 进程
        proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    except FileNotFoundError:
        print("[hobot_vision_node] Error: 'bash' command not found.")
        sys.exit(1)

    # 捕获退出信号，优雅地关闭所有后台子进程
    def handler(signum, frame):
        print("\n[hobot_vision_node] Stopping vision pipeline...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # 阻塞等待
    proc.wait()

if __name__ == '__main__':
    main()
