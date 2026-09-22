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
from services.action.action_command import ACTION_REQUEST_TOPIC
import evdev
from evdev import ecodes

from services.motion.joystick_axis import normalize_axis_value
from services.motion.joystick_servo_targets import (
    AXIS_L2,
    AXIS_R2,
    AXIS_RX,
    AXIS_RY,
    compute_joystick_servo_targets,
)
from services.motion.motor_control import mix_differential_drive
from services.motion.motion_arbiter import MOTOR_JOYSTICK_TOPIC, STOP_COMMAND
from services.motion.remote_control_config import RemoteControlConfigWatcher
from services.motion.servo_motion_config import load_neck_kinematics
from services.game.game_protocol import (
    GAME_MODE_REQUEST_TOPIC,
    GAME_MODE_STATE_TOPIC,
    encode_game_request,
    game_is_active,
)
from services.motion.joystick_button_policy import (
    BTN_A,
    BTN_B,
    BTN_L1,
    BTN_R1,
    BTN_X,
    BTN_Y,
    DEFAULT_AUTO_RESET_DELAY,
    HAT_X,
    HAT_Y,
    JoystickButtonPolicy,
)

# --- 按键/轴映射 ---
AXIS_LX = 0  # 左摇杆 X
AXIS_LY = 1  # 左摇杆 Y

# 倒计时设置
AUTO_RESET_DELAY = DEFAULT_AUTO_RESET_DELAY

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

        self.action_pub = self.create_publisher(String, ACTION_REQUEST_TOPIC, 10)
        self.motor_pub = self.create_publisher(String, MOTOR_JOYSTICK_TOPIC, 10)
        self.game_request_pub = self.create_publisher(String, GAME_MODE_REQUEST_TOPIC, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)

        self.device = None
        self.running = False
        self._scan_thread = None
        self._game_active = False
        self._was_moving = False
        self._button_policy = JoystickButtonPolicy(hold_seconds=2.0)

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
                    
                    if code in (HAT_X, HAT_Y):
                        decision = self._button_policy.handle_hat(
                            code,
                            val,
                            now=time.time(),
                            auto_reset_delay=AUTO_RESET_DELAY,
                        )
                        self._apply_button_decision(decision)
                    elif code in self._axes:
                        info = None
                        for c, a in self.device.capabilities(verbose=False).get(3, []):
                            if c == code:
                                info = a
                                break
                        if info:
                            self._axes[code] = normalize_axis_value(
                                val,
                                info.min,
                                info.max,
                                deadzone=self.deadzone,
                                is_trigger=code in (AXIS_L2, AXIS_R2),
                                invert_y=code in (AXIS_LY, AXIS_RY),
                            )

                elif event.type == ecodes.EV_KEY:
                    decision = self._button_policy.handle_key(
                        event.code,
                        event.value,
                        now=time.time(),
                        auto_reset_delay=AUTO_RESET_DELAY,
                        game_active=self._game_active,
                    )
                    self._apply_button_decision(decision)

        except OSError:
            pass

    def _apply_button_decision(self, decision):
        self._auto_timers.update(decision.timer_updates)
        if decision.action:
            self._send_action_cmd(decision.action)

    def _tick_loop(self):
        self._refresh_remote_config()
        if self.device is None: return
        if self._button_policy.poll_game_toggle(game_active=self._game_active):
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
            if self._was_moving:
                self._stop_motors()
                self._was_moving = False
        else:
            self._was_moving = True
            cmd_motor = mix_differential_drive(ly, lx)
            msg_m = String()
            msg_m.data = json.dumps(cmd_motor)
            self.motor_pub.publish(msg_m)

        # 2. 结算舵机指令 manual_servo (右摇杆、扳机、自动复位计时器)
        targets = compute_joystick_servo_targets(
            self._axes,
            self._auto_timers,
            now,
            self._neck_kinematics,
        )

        # 发送 manual_servo
        msg_s = String()
        msg_s.data = json.dumps({
            "name": "manual_servo", 
            "arguments": {"targets": targets, "step_size": self.servo_step_size},
            "source": "joystick",
        }, ensure_ascii=False)
        self.action_pub.publish(msg_s)

    def _send_action_cmd(self, name, args=None):
        payload = {"name": name}
        if args: payload["arguments"] = args
        payload["source"] = "joystick"
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
