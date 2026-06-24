#!/usr/bin/env python3
# nodes/action_ros_node.py
# MCP tool_call -> ROS 话题 的动作分发器
# 订阅 /action_cmd，翻译到 /servo_cmd、/motor_cmd、/tft_cmd

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ActionRosNode(Node):
    # ── 动作 -> 舵机映射 ──
    ACTION_TO_SERVO = {
        "dance": [
            {"name": "arm_r", "angle": 140}, {"name": "arm_l", "angle": 40},
            {"name": "head_yaw", "angle": 110},
        ],
        "talk_micro_move": [
            {"name": "arm_r", "angle": 95}, {"name": "arm_l", "angle": 85},
        ],
        "wave":            [{"name": "arm_r", "angle": 140}, {"name": "arm_l", "angle": 140}],
        "nod":             [{"name": "neck_bottom", "angle": 100}],
        "shake_head":      [{"name": "head_yaw", "angle": 65}],
        "look_up":         [{"name": "neck_bottom", "angle": 110}],
        "look_down":       [{"name": "neck_bottom", "angle": 70}],
        "tilt_head":       [{"name": "neck_top", "angle": 115}],
    }

    # ── 移动方向 -> 电机指令 ──
    MOTION_TO_MOTOR = {
        "forward":  {"left": {"action": 1, "throttle": 55}, "right": {"action": 1, "throttle": 55}},
        "backward": {"left": {"action": 2, "throttle": 55}, "right": {"action": 2, "throttle": 55}},
        "spin":     {"left": {"action": 2, "throttle": 55}, "right": {"action": 1, "throttle": 55}},
        "left":     {"left": {"action": 2, "throttle": 45}, "right": {"action": 1, "throttle": 55}},
        "right":    {"left": {"action": 1, "throttle": 55}, "right": {"action": 2, "throttle": 45}},
    }

    def __init__(self):
        super().__init__('action_ros_node')

        # 发布舵机/电机/TFT 指令
        self.servo_pub = self.create_publisher(String, '/servo_cmd', 10)
        self.motor_pub = self.create_publisher(String, '/motor_cmd', 10)
        self.tft_pub   = self.create_publisher(String, '/tft_cmd', 10)

        # Oneshot 定时器
        self._servo_timer = None
        self._motor_timer = None

        # 订阅 /action_cmd
        self.sub_action = self.create_subscription(
            String, '/action_cmd', self.action_callback, 10)

        self.get_logger().info('Action ROS node online, listening on /action_cmd')

    def action_callback(self, msg):
        try:
            cmd_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"/action_cmd JSON parse failed: {msg.data}")
            return

        tool_name = cmd_data.get("name")
        args_obj = cmd_data.get("arguments", {})
        if isinstance(args_obj, str):
            args = json.loads(args_obj or "{}")
        else:
            args = args_obj

        if tool_name == "express_emotion":
            emotion = args.get("emotion", "happy")
            msg_out = String()
            msg_out.data = f"eyeaction:{emotion}\n"
            self.tft_pub.publish(msg_out)
            self.get_logger().info(f'[Action] TFT eyeaction:{emotion}')

        elif tool_name == "perform_action":
            action = args.get("action", "talk_micro_move")
            servo_list = self.ACTION_TO_SERVO.get(action)
            if servo_list:
                for cmd in servo_list:
                    msg_out = String()
                    msg_out.data = json.dumps(cmd, ensure_ascii=False)
                    self.servo_pub.publish(msg_out)
                    self.get_logger().info(
                        f'[Action] servo {cmd["name"]} -> {cmd["angle"]} deg'
                    )
                # 1s 后回正 (oneshot)
                self._servo_timer = self.create_timer(
                    1.0, lambda s=servo_list: self._reset_servos(s)
                )
            else:
                self.get_logger().warn(f'[Action] unknown perform_action: {action}')

        elif tool_name == "move_chassis":
            direction = args.get("direction", "forward")
            duration = float(args.get("duration", 1))
            motor = self.MOTION_TO_MOTOR.get(direction)
            if motor:
                msg_out = String()
                msg_out.data = json.dumps(motor, ensure_ascii=False)
                self.motor_pub.publish(msg_out)
                self.get_logger().info(f'[Action] chassis {direction} for {duration}s')
                self._motor_timer = self.create_timer(duration, self._stop_motors)
            else:
                self.get_logger().warn(f'[Action] unknown direction: {direction}')

        else:
            self.get_logger().warn(f'[Action] unhandled tool: {tool_name}')

    def _reset_servos(self, servo_list):
        for cmd in servo_list:
            msg_out = String()
            msg_out.data = json.dumps({"name": cmd["name"], "angle": 90}, ensure_ascii=False)
            self.servo_pub.publish(msg_out)
        if self._servo_timer is not None:
            self.destroy_timer(self._servo_timer)
            self._servo_timer = None

    def _stop_motors(self):
        stop_cmd = {
            "left": {"action": 0, "throttle": 0},
            "right": {"action": 0, "throttle": 0}
        }
        msg_out = String()
        msg_out.data = json.dumps(stop_cmd, ensure_ascii=False)
        self.motor_pub.publish(msg_out)
        self.get_logger().info('[Action] motors stopped')
        if self._motor_timer is not None:
            self.destroy_timer(self._motor_timer)
            self._motor_timer = None

    def destroy_node(self):
        self.get_logger().info('Closing action node...')
        if self._motor_timer is not None:
            self.destroy_timer(self._motor_timer)
            self._motor_timer = None
        if self._servo_timer is not None:
            self.destroy_timer(self._servo_timer)
            self._servo_timer = None
        self._stop_motors()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ActionRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
