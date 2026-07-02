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
    
    cmd = [
        "bash", "-c",
        "source /opt/tros/humble/setup.bash && ros2 launch dnn_node_example dnn_node_example.launch.py"
    ]
    
    print(f"[hobot_vision_node] Starting: {cmd[2]} with CAM_TYPE=usb")
    
    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError:
        print("[hobot_vision_node] Error: 'ros2' command not found. Are you on the Sunrise Pi with TROS sourced?")
        sys.exit(1)

    # 捕获退出信号，优雅地关闭子进程
    def handler(signum, frame):
        print("\n[hobot_vision_node] Stopping ros2 launch...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    # 阻塞等待
    proc.wait()

if __name__ == '__main__':
    main()
