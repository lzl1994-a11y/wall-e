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

def main():
    env = os.environ.copy()
    # 注入 USB 摄像头环境变量
    env["CAM_TYPE"] = "usb"
    # 手动构建稳定的 TROS 视觉管线 (兼容 MJPEG USB 摄像头)
    # 1. usb_cam: 读取 USB 摄像头，默认发布 MJPEG 格式到 /image
    # 2. hobot_codec: 订阅 /image (MJPEG)，硬件解码为 NV12 发布到 /image_nv12
    # 3. mono2d: 切换到模型目录运行，订阅 /image_nv12，发布 AI 框到 /hobot_mono2d_body_detection
    # 4. websocket: 订阅 /image (MJPEG) 和 AI 框，开启 8000 端口网页服务
    cmd = [
        "bash", "-c",
        "source /opt/tros/humble/setup.bash && "
        "ros2 run hobot_usb_cam hobot_usb_cam --ros-args -p video_device:=/dev/video0 -p image_width:=960 -p image_height:=544 & "
        "ros2 run hobot_codec hobot_codec_republish --ros-args -p channel:=1 -p in_mode:=ros -p in_format:=jpeg -p out_mode:=ros -p out_format:=nv12 -p sub_topic:=/image -p pub_topic:=/image_nv12 & "
        "(cd /opt/tros/humble/lib/mono2d_body_detection && ros2 run mono2d_body_detection mono2d_body_detection --ros-args -p is_shared_mem_sub:=0 -p ros_img_topic_name:=/image_nv12 -p ai_msg_pub_topic_name:=/hobot_mono2d_body_detection) & "
        "ros2 run websocket websocket --ros-args -p image_topic:=/image -p image_type:=mjpeg -p msg_pub_topic_name:=/hobot_mono2d_body_detection & "
        "wait"
    ]
    
    print(f"[hobot_vision_node] Starting robust manual vision pipeline...")
    
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
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # 阻塞等待
    proc.wait()

if __name__ == '__main__':
    main()
