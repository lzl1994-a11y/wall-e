#!/usr/bin/env python3
"""
手柄遥控节点（高阶纯手工映射版）
平台：Ubuntu / 旭日X3派 (依赖 evdev)
"""

import time
import json
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import evdev
from evdev import ecodes

# --- 按键/轴映射 ---
AXIS_LX = 0  # 左摇杆 X
AXIS_LY = 1  # 左摇杆 Y
AXIS_RX = 3  # 右摇杆 X
AXIS_RY = 4  # 右摇杆 Y
AXIS_L2 = 2  # LT 扳机
AXIS_R2 = 5  # RT 扳机
HAT_X = 16   # 十字键 X
HAT_Y = 17   # 十字键 Y

BTN_L1 = 310
BTN_R1 = 311
BTN_A = 304
BTN_B = 305
BTN_X = 307
BTN_Y = 308

# 倒计时设置
AUTO_RESET_DELAY = 3.0

class JoyControlNode(Node):
    def __init__(self):
        super().__init__("joy_control_node")

        self.action_pub = self.create_publisher(String, '/action_cmd', 10)
        self.motor_pub = self.create_publisher(String, '/motor_cmd', 10)

        self.device = None
        self.running = False
        self._scan_thread = None

        # 模拟轴归一化状态 (-1.0 到 1.0, 扳机为 0.0 到 1.0)
        self._axes = {
            AXIS_LX: 0.0, AXIS_LY: 0.0,
            AXIS_RX: 0.0, AXIS_RY: 0.0,
            AXIS_L2: 0.0, AXIS_R2: 0.0
        }
        self.deadzone = 0.15
        
        # 计时器状态
        self._timers = {
            'arm_l': 0.0, 'arm_r': 0.0,
            'eyebrow_l': 0.0, 'eyebrow_r': 0.0
        }

        self._motor_publish_timer = self.create_timer(0.05, self._tick_loop) # 20Hz 极高频刷新

        self.get_logger().info("手柄节点启动，等待手柄连接...")
        self._start_scanning()

    def _start_scanning(self):
        self.running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()

    def _find_device(self):
        for p in evdev.list_devices():
            try:
                d = evdev.InputDevice(p)
                caps = d.capabilities(verbose=False)
                if caps.get(3) and caps.get(1):
                    return d
            except: pass
        return None

    def _scan_loop(self):
        while self.running and rclpy.ok():
            if self.device is None:
                dev = self._find_device()
                if dev:
                    self.get_logger().info(f"手柄连接: {dev.name}")
                    self.device = dev
                    self._send_action_cmd("set_tracking_mode", {"mode": "idle"})
                    self._run_control()
                    self.get_logger().info("手柄断开。")
                    self._stop_motors()
                    self.device = None
                    try: dev.close()
                    except: pass
            time.sleep(1.0)

    def _run_control(self):
        try:
            for event in self.device.read_loop():
                if not self.running: break

                if event.type == ecodes.EV_ABS:
                    code = event.code
                    val = event.value
                    
                    if code == HAT_X:
                        if val == -1: # 左
                            self._timers['arm_l'] = time.time() + AUTO_RESET_DELAY
                        elif val == 1: # 右
                            self._timers['arm_r'] = time.time() + AUTO_RESET_DELAY
                    elif code == HAT_Y:
                        if val == -1: # 上
                            self._timers['arm_l'] = time.time() + AUTO_RESET_DELAY
                            self._timers['arm_r'] = time.time() + AUTO_RESET_DELAY
                        elif val == 1: # 下
                            self._timers['arm_l'] = 0.0
                            self._timers['arm_r'] = 0.0
                    elif code in self._axes:
                        # 归一化
                        info = None
                        for c, a in self.device.capabilities(verbose=False).get(3, []):
                            if c == code:
                                info = a
                                break
                        if info:
                            if code in (AXIS_L2, AXIS_R2):
                                # 扳机 (0 ~ 255) -> 0.0 ~ 1.0
                                n_val = max(0, val - info.min) / max(1, info.max - info.min)
                                self._axes[code] = n_val
                            else:
                                # 摇杆 (-32768 ~ 32767) -> -1.0 ~ 1.0
                                mid = (info.min + info.max) / 2.0
                                n_val = (val - mid) / float(info.max - mid)
                                if abs(n_val) < self.deadzone: n_val = 0.0
                                # Y轴翻转，让上推变为正
                                if code in (AXIS_LY, AXIS_RY):
                                    n_val = -n_val
                                self._axes[code] = n_val

                elif event.type == ecodes.EV_KEY:
                    if event.value == 1: # 按下
                        if event.code == BTN_L1:
                            self._timers['eyebrow_l'] = time.time() + AUTO_RESET_DELAY
                        elif event.code == BTN_R1:
                            self._timers['eyebrow_r'] = time.time() + AUTO_RESET_DELAY
                        elif event.code == BTN_A: self._send_action_cmd("happy_dance")
                        elif event.code == BTN_B: self._send_action_cmd("sad_react")
                        elif event.code == BTN_X: self._send_action_cmd("wave_hello")
                        elif event.code == BTN_Y: self._send_action_cmd("raise_hand")

        except OSError:
            pass

    def _tick_loop(self):
        if self.device is None: return
        now = time.time()

        # 1. 结算电机底盘 (左摇杆: LY前进, LX转向)
        ly = self._axes[AXIS_LY]
        lx = self._axes[AXIS_LX]
        
        if ly == 0.0 and lx == 0.0:
            if getattr(self, '_was_moving', False):
                self._stop_motors()
                self._was_moving = False
        else:
            self._was_moving = True
            left_speed = ly + lx
            right_speed = ly - lx
            left_speed = max(min(left_speed, 1.0), -1.0)
            right_speed = max(min(right_speed, 1.0), -1.0)
            
            cmd_motor = {
                "left": {"action": 1 if left_speed > 0 else (2 if left_speed < 0 else 0), "throttle": int(abs(left_speed) * 100)},
                "right": {"action": 1 if right_speed > 0 else (2 if right_speed < 0 else 0), "throttle": int(abs(right_speed) * 100)}
            }
            msg_m = String()
            msg_m.data = json.dumps(cmd_motor)
            self.motor_pub.publish(msg_m)

        # 2. 结算舵机指令 manual_servo (右摇杆、扳机、自动复位计时器)
        targets = {}
        
        # 头部方向 (右摇杆)
        rx = self._axes[AXIS_RX] # 左:-1, 右:1
        ry = self._axes[AXIS_RY] # 上:1, 下:-1
        
        targets['head_yaw'] = int(5000 - rx * 2600) # rx=1(右) -> 1920, rx=-1(左) -> 7600
        
        # 脖子俯仰
        # ry=1(上) -> neck_top=4000, neck_bottom=7000
        # ry=-1(下) -> neck_top=6000, neck_bottom=3500
        targets['neck_top'] = int(5000 - ry * 1000)
        if ry > 0:
            targets['neck_bottom'] = int(4000 + ry * 3000)
        else:
            targets['neck_bottom'] = int(4000 + ry * 500) # ry是负数

        # 眼睛扳机 (L2/R2: 0.0 ~ 1.0)
        l2 = self._axes[AXIS_L2]
        r2 = self._axes[AXIS_R2]
        targets['eye_l'] = int(7500 - l2 * 2500) # 0->7500, 1->5000
        targets['eye_r'] = int(2000 + r2 * 2000) # 0->2000, 1->4000

        # 手臂与眉毛 (倒计时逻辑)
        targets['arm_l'] = 6000 if now < self._timers['arm_l'] else 2000
        targets['arm_r'] = 4000 if now < self._timers['arm_r'] else 8000
        targets['eyebrow_l'] = 5700 if now < self._timers['eyebrow_l'] else 8000
        targets['eyebrow_r'] = 4200 if now < self._timers['eyebrow_r'] else 1920

        # 发送 manual_servo
        msg_s = String()
        msg_s.data = json.dumps({
            "name": "manual_servo", 
            "arguments": {"targets": targets, "step_size": 40.0}
        }, ensure_ascii=False)
        self.action_pub.publish(msg_s)

    def _send_action_cmd(self, name, args=None):
        payload = {"name": name}
        if args: payload["arguments"] = args
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.action_pub.publish(msg)

    def _stop_motors(self):
        msg = String()
        msg.data = json.dumps({"left": {"action": 0, "throttle": 0}, "right": {"action": 0, "throttle": 0}})
        self.motor_pub.publish(msg)

    def shutdown(self):
        self.running = False
        self._stop_motors()
        if self.device:
            try: self.device.close()
            except: pass

def main(args=None):
    rclpy.init(args=args)
    node = JoyControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()