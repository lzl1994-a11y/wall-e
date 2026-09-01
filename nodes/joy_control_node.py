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

from services.motor_control import mix_differential_drive
from services.motion_arbiter import MOTOR_JOYSTICK_TOPIC, STOP_COMMAND
from services.remote_control_config import RemoteControlConfigWatcher
from services.servo_motion_config import load_neck_kinematics
from services.game_hotkey import ButtonChordHold
from services.game_protocol import (
    GAME_MODE_REQUEST_TOPIC,
    GAME_MODE_STATE_TOPIC,
    encode_game_request,
    game_is_active,
)

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

        self._remote_config_watcher = RemoteControlConfigWatcher()
        remote_config = self._remote_config_watcher.load_if_changed()
        if remote_config is None:
            raise RuntimeError("首次读取手柄遥控配置失败")
        self.servo_step_size = remote_config["servo_step_size"]
        self.update_rate_hz = remote_config["update_rate_hz"]
        self._neck_kinematics = load_neck_kinematics()

        self.action_pub = self.create_publisher(String, '/action_cmd', 10)
        self.motor_pub = self.create_publisher(String, MOTOR_JOYSTICK_TOPIC, 10)
        self.game_request_pub = self.create_publisher(String, GAME_MODE_REQUEST_TOPIC, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)

        self.device = None
        self.running = False
        self._scan_thread = None
        self._game_active = False
        self._game_hotkey = ButtonChordHold(hold_seconds=2.0)
        self._game_hotkey_fired = False
        self._x_down = False
        self._y_down = False

        # 模拟轴归一化状态 (-1.0 到 1.0, 扳机为 0.0 到 1.0)
        self._axes = {
            AXIS_LX: 0.0, AXIS_LY: 0.0,
            AXIS_RX: 0.0, AXIS_RY: 0.0,
            AXIS_L2: 0.0, AXIS_R2: 0.0
        }
        self.deadzone = 0.15
        
        # 计时器状态
        self._auto_timers = {
            'arm_l': 0.0, 'arm_r': 0.0,
            'eyebrow_l': 0.0, 'eyebrow_r': 0.0
        }

        self._motor_publish_timer = self.create_timer(1.0 / self.update_rate_hz, self._tick_loop)

        self.get_logger().info(
            f"手柄节点启动，等待手柄连接... "
            f"(舵机步长={self.servo_step_size:g}, 更新频率={self.update_rate_hz:g}Hz)"
        )
        self._start_scanning()

    def _refresh_remote_config(self):
        remote_config = self._remote_config_watcher.load_if_changed()
        if remote_config is None:
            return

        new_step_size = remote_config["servo_step_size"]
        new_update_rate_hz = remote_config["update_rate_hz"]
        step_changed = new_step_size != self.servo_step_size
        rate_changed = new_update_rate_hz != self.update_rate_hz
        if not step_changed and not rate_changed:
            return

        self.servo_step_size = new_step_size
        if rate_changed:
            old_timer = self._motor_publish_timer
            self.update_rate_hz = new_update_rate_hz
            self._motor_publish_timer = self.create_timer(
                1.0 / self.update_rate_hz, self._tick_loop
            )
            self.destroy_timer(old_timer)

        self.get_logger().info(
            f"手柄遥控配置已热更新 "
            f"(舵机步长={self.servo_step_size:g}, 更新频率={self.update_rate_hz:g}Hz)"
        )

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
                    self.game_request_pub.publish(
                        String(data=encode_game_request("controller_disconnected"))
                    )
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
                            self._auto_timers['arm_l'] = time.time() + AUTO_RESET_DELAY
                        elif val == 1: # 右
                            self._auto_timers['arm_r'] = time.time() + AUTO_RESET_DELAY
                    elif code == HAT_Y:
                        if val == -1: # 上
                            self._auto_timers['arm_l'] = time.time() + AUTO_RESET_DELAY
                            self._auto_timers['arm_r'] = time.time() + AUTO_RESET_DELAY
                        elif val == 1: # 下
                            self._auto_timers['arm_l'] = 0.0
                            self._auto_timers['arm_r'] = 0.0
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
                    if event.code in {BTN_X, BTN_Y} and event.value in {0, 1}:
                        down = event.value == 1
                        if event.code == BTN_X:
                            self._x_down = down
                            self._game_hotkey.set_first(down)
                        else:
                            self._y_down = down
                            self._game_hotkey.set_second(down)
                        other_down = self._y_down if event.code == BTN_X else self._x_down
                        if (
                            not down
                            and not other_down
                            and not self._game_hotkey_fired
                            and not self._game_active
                        ):
                            if event.code == BTN_X:
                                self._send_action_cmd("wave_hello")
                            else:
                                self._send_action_cmd("raise_hand")
                        if not self._x_down and not self._y_down:
                            self._game_hotkey_fired = False
                        continue

                    if self._game_active:
                        continue

                    if event.value == 1: # 按下
                        if event.code == BTN_L1:
                            self._auto_timers['eyebrow_l'] = time.time() + AUTO_RESET_DELAY
                        elif event.code == BTN_R1:
                            self._auto_timers['eyebrow_r'] = time.time() + AUTO_RESET_DELAY
                        elif event.code == BTN_A: self._send_action_cmd("happy_dance")
                        elif event.code == BTN_B: self._send_action_cmd("sad_react")

        except OSError:
            pass

    def _tick_loop(self):
        self._refresh_remote_config()
        if self.device is None: return
        if not self._game_active and self._game_hotkey.poll():
            self._game_hotkey_fired = True
            self._stop_motors()
            self.game_request_pub.publish(String(data=encode_game_request(
                "toggle", controller=getattr(self.device, "path", "/dev/input/event2")
            )))
            return
        if self._game_active:
            return
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
            cmd_motor = mix_differential_drive(ly, lx)
            msg_m = String()
            msg_m.data = json.dumps(cmd_motor)
            self.motor_pub.publish(msg_m)

        # 2. 结算舵机指令 manual_servo (右摇杆、扳机、自动复位计时器)
        targets = {}
        
        # 头部方向 (右摇杆)
        rx = self._axes[AXIS_RX] # 左:-1, 右:1
        ry = self._axes[AXIS_RY] # 上:1, 下:-1
        
        targets['head_yaw'] = int(5000 - rx * 2600) # rx=1(右) -> 1920, rx=-1(左) -> 7600
        
        # 脖子俯仰：中心、上下限和双舵机联动均来自 config.yaml。
        targets.update(self._neck_kinematics.targets(ry))

        # 眼睛扳机 (L2/R2: 0.0 ~ 1.0)
        l2 = self._axes[AXIS_L2]
        r2 = self._axes[AXIS_R2]
        targets['eye_l'] = int(7500 - l2 * 2500) # 0->7500, 1->5000
        targets['eye_r'] = int(2000 + r2 * 2000) # 0->2000, 1->4000

        # 手臂与眉毛 (倒计时逻辑)
        targets['arm_l'] = 6000 if now < self._auto_timers['arm_l'] else 2000
        targets['arm_r'] = 4000 if now < self._auto_timers['arm_r'] else 8000
        targets['eyebrow_l'] = 5700 if now < self._auto_timers['eyebrow_l'] else 8000
        targets['eyebrow_r'] = 4200 if now < self._auto_timers['eyebrow_r'] else 1920

        # 发送 manual_servo
        msg_s = String()
        msg_s.data = json.dumps({
            "name": "manual_servo", 
            "arguments": {"targets": targets, "step_size": self.servo_step_size}
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
        msg.data = json.dumps(STOP_COMMAND)
        self.motor_pub.publish(msg)

    def _on_game_state(self, message):
        active = game_is_active(message.data)
        if active and not self._game_active:
            self._stop_motors()
            for axis in self._axes:
                self._axes[axis] = 0.0
        self._game_active = active

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
