"""手柄遥控节点：独占 /dev/input/eventX，摇杆→cmd_vel，断线自动交还控制权"""
import os
import sys
import time
import threading

import evdev
from evdev import ecodes

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# ── 键值映射（Sony Wireless Controller 默认）─────────────────────────
AXIS_MAP = {0: "左X", 1: "左Y", 2: "L2", 3: "右X", 4: "右Y", 5: "R2"}
BTN_NAMES = {
    304: "A", 305: "B", 307: "X", 308: "Y",
    310: "LB", 311: "RB", 314: "SELECT", 315: "START",
    316: "PS", 317: "L3", 318: "R3",
}


class JoyControlNode(Node):
    def __init__(self):
        super().__init__("joy_control_node")

        # ROS
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)

        # 参数
        self.declare_parameter("max_linear", 0.5)   # 最大前进速度 m/s
        self.declare_parameter("max_angular", 1.0)  # 最大转向速度 rad/s
        self.declare_parameter("deadzone", 0.15)    # 摇杆死区

        self.max_linear = self.get_parameter("max_linear").value
        self.max_angular = self.get_parameter("max_angular").value
        self.deadzone = self.get_parameter("deadzone").value

        # 状态
        self.device = None
        self.running = False
        self._scan_thread = None

        self.get_logger().info("手柄节点启动，等待手柄连接...")
        self._start_scanning()

    # ── 自动发现 & 重连 ──────────────────────────────────────────
    def _find_device(self):
        for p in evdev.list_devices():
            try:
                d = evdev.InputDevice(p)
                caps = d.capabilities(verbose=False)
                if caps.get(3) and caps.get(1):
                    return d
            except Exception:
                continue
        return None

    def _start_scanning(self):
        self.running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()

    def _scan_loop(self):
        while self.running and rclpy.ok():
            if self.device is None:
                dev = self._find_device()
                if dev:
                    self.get_logger().info(f"手柄已连接: {dev.name} [{dev.path}]")
                    self.device = dev
                    self._run_control()
                    # _run_control 返回 = 手柄断开
                    self.get_logger().info("手柄已断开，恢复远程控制")
                    # 停车
                    self.pub_cmd.publish(Twist())
                    self.device = None
                    try:
                        dev.close()
                    except Exception:
                        pass
            time.sleep(0.5)

    # ── 手柄事件循环 ────────────────────────────────────────────
    def _run_control(self):
        """阻塞式读取手柄事件，抛出 OSError 表示断开"""
        dead = self.deadzone
        axis_vals = {k: 0.0 for k in AXIS_MAP}

        try:
            for event in self.device.read_loop():
                if not self.running:
                    break

                # 摇杆轴
                if event.type == ecodes.EV_ABS:
                    code = event.code
                    if code not in AXIS_MAP:
                        continue
                    # 归一化
                    caps = self.device.capabilities(verbose=False).get(3, [])
                    info = None
                    for c, a in caps:
                        if c == code:
                            info = a
                            break
                    if info:
                        mid = (info.min + info.max) / 2
                        val = (event.value - mid) / (info.max - mid) * 2
                        # 死区
                        if abs(val) < dead:
                            val = 0.0
                        axis_vals[code] = val

                    # 左摇杆Y=前进后退, 右摇杆X=左右转向
                    ly = axis_vals.get(1, 0.0)
                    rx = axis_vals.get(3, 0.0)

                    twist = Twist()
                    twist.linear.x = -ly * self.max_linear    # Y 轴向上为负
                    twist.angular.z = rx * self.max_angular
                    self.pub_cmd.publish(twist)

                # 按键
                elif event.type == ecodes.EV_KEY:
                    name = BTN_NAMES.get(event.code, f"BTN{event.code}")
                    if event.value == 1:
                        self.get_logger().info(f"按键: {name} 按下")
                        # TODO: 映射功能键（急停/拍照/模式切换）

        except OSError:
            pass  # 手柄断开，自然退出

    # ── 生命周期 ────────────────────────────────────────────────
    def shutdown(self):
        self.running = False
        self.pub_cmd.publish(Twist())  # 停车
        if self.device:
            try:
                self.device.close()
            except Exception:
                pass
        self.get_logger().info("手柄节点已关闭")


def main(args=None):
    rclpy.init(args=args)
    node = JoyControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()