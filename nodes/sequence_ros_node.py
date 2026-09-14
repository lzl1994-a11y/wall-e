#!/usr/bin/env python3
# nodes/sequence_ros_node.py
# 统一轨迹控制器：接收仲裁后的 /action_cmd，支持单一动作与成组动作 (Timeline)
import time
import json
from services.action_cancel import ACTION_CANCEL_TOPIC, parse_action_cancel
from services.action_command import ACTION_COMMAND_TOPIC, parse_action_request
from services.action_status import ACTION_STATUS_TOPIC, build_action_status
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from services.motion_arbiter import MOTOR_AUTONOMY_TOPIC, STOP_COMMAND
from services.vision_pipeline_protocol import TRACKING_SERVO_TARGET_TOPIC
from services.dialog_expression_protocol import DIALOG_EXPRESSION_TARGET_TOPIC
from services.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active
from services.sequence_execution import (
    SequenceLibrary,
    ServoTrajectory,
    load_yaml_mapping,
)

class SequenceRosNode(Node):
    # 所有的动作预设已迁移至 sequences.yaml，由 _flatten_sequence 处理

    MOTION_TO_MOTOR = {
        "forward":  {"left": {"action": 1, "throttle": 55}, "right": {"action": 1, "throttle": 55}},
        "backward": {"left": {"action": 2, "throttle": 55}, "right": {"action": 2, "throttle": 55}},
        "spin":     {"left": {"action": 2, "throttle": 55}, "right": {"action": 1, "throttle": 55}},
        "left":     {"left": {"action": 2, "throttle": 45}, "right": {"action": 1, "throttle": 55}},
        "right":    {"left": {"action": 1, "throttle": 55}, "right": {"action": 2, "throttle": 45}},
    }

    def __init__(self):
        super().__init__('sequence_ros_node')
        
        # 1. 加载配置
        config = self._load_yaml('core/config.yaml')
        servos_list = config.get('servos', [])
        # 转成 dict 方便快速查找
        self._servos_config = {s['name']: s for s in servos_list}
        
        seq_yaml = self._load_yaml('core/sequences.yaml')
        self._sequences = seq_yaml.get('sequences', {})
        self._poses = seq_yaml.get('poses', {})

        # 2. ROS-independent sequence and servo execution services.  The
        # aliases preserve the existing node-level diagnostics and tests while
        # state ownership moves into the service layer.
        self._sequence_library = SequenceLibrary(
            self._sequences,
            self._poses,
            on_error=self.get_logger().error,
        )
        self._trajectory = ServoTrajectory(self._servos_config)
        self._servos_config = self._trajectory.servos
        self._virtual_state = self._trajectory.virtual_state
        self._targets = self._trajectory.targets
        self._steps = self._trajectory.steps

        # 时间轴与队列
        self._current_sequence = []
        self._sequence_start_time = 0.0
        self._active_motor_cmd = None
        self._motor_stop_at = 0.0
        self._motor_request = None
        self._sequence_request = None
        self._auto_reset_timer = None
        self._game_active = False
        self._explicit_motion_active = False
        self._pending_dialog_expression = None

        # 3. ROS 接口
        self.servo_pub = self.create_publisher(String, '/servo_cmd', 10)
        self.motor_pub = self.create_publisher(String, MOTOR_AUTONOMY_TOPIC, 10)
        self.tft_pub   = self.create_publisher(String, '/tft_cmd', 10)
        self.action_status_pub = self.create_publisher(String, ACTION_STATUS_TOPIC, 10)

        # 4. 核心 50Hz 插值定时器
        self.create_timer(0.02, self._tick)

        # 统一订阅 /action_cmd，负责动作编排和运动指令分发
        self.create_subscription(String, ACTION_COMMAND_TOPIC, self._on_action_cmd, 10)
        self.create_subscription(String, ACTION_CANCEL_TOPIC, self._on_action_cancel, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        # Tracking produces targets at detector frame rate. Depth 1 makes this
        # a latest-value stream and avoids replaying stale head positions.
        self.create_subscription(
            String,
            TRACKING_SERVO_TARGET_TOPIC,
            self._on_tracking_servo_targets,
            1,
        )
        self.create_subscription(
            String,
            DIALOG_EXPRESSION_TARGET_TOPIC,
            self._on_dialog_expression_targets,
            1,
        )
        
        self.get_logger().info(
            f'Sequence ROS Node online, consuming {ACTION_COMMAND_TOPIC}. '
            '50Hz interpolation running.'
        )
        
    def _load_yaml(self, path):
        return load_yaml_mapping(path, on_error=self.get_logger().error)

    def _clamp_pwm(self, name, raw_pwm):
        """将传入的原始 PWM 值限制在安全的硬件限位内"""
        return self._trajectory.clamp(name, raw_pwm)

    def _servo_init(self, name, fallback):
        return self._trajectory.initial(name, fallback)

    def _on_action_cmd(self, msg):
        if self._game_active:
            return
        request = parse_action_request(msg.data)
        if request is None:
            return
        tool = request["name"]
        args = request["arguments"]
        request_id = request.get("request_id")

        # Tracking and camera tools have separate owners. Ignoring them here
        # also prevents an unrelated tool call from interrupting a sequence.
        if tool not in {
            "express_emotion", "move_chassis", "manual_servo",
            "play_sequence", "stop_all",
        }:
            return

        # ===== 外部打断机制核心：清空队列，并清零步长 =====
        self._interrupt_sequence("superseded_by_new_command")
        if self._active_motor_cmd is not None:
            self._stop_motors(status="interrupted", detail="superseded_by_new_command")
        self._current_sequence = [] # 打断成组动作
        for name in self._steps:
            self._steps[name] = 0.0 # 清零步长，平滑运动瞬间停止
        if self._auto_reset_timer:
            self.destroy_timer(self._auto_reset_timer)
            self._auto_reset_timer = None
        self.get_logger().info(f"[Interrupt] Cleared state for tool: {tool}")

        # ===== 指令分发 =====
        if tool == "express_emotion":
            self._publish_request_status(request, "accepted")
            self._dispatch_action({"type": "express_emotion", "emotion": args.get("emotion", "happy")})
            self._publish_request_status(request, "completed")
            
        elif tool == "move_chassis":
            direction = args.get("direction", "")
            if direction not in self.MOTION_TO_MOTOR:
                self._publish_request_status(request, "rejected", "invalid_direction")
                return
            self._motor_request = request if request_id else None
            self._publish_request_status(request, "accepted")
            self._dispatch_action({
                "type": "motor", 
                "direction": direction,
                "duration": float(args.get("duration", 1.0))
            })
            
        elif tool == "manual_servo":
            self._explicit_motion_active = True
            self._publish_request_status(request, "accepted")
            self._dispatch_action({
                "type": "manual_servo",
                "targets": args.get("targets", {}),
                "step_size": args.get("step_size", 30.0)
            })
            self._publish_request_status(request, "completed")
            

        elif tool == "play_sequence":
            seq_name = args.get("sequence_name", "")
            
            # 使用时间轴扁平化算法拆解嵌套序列
            flattened_frames = self._flatten_sequence(seq_name, offset_time=0.0)
            if flattened_frames:
                self._explicit_motion_active = True
                # 按照绝对时间进行排序
                flattened_frames.sort(key=lambda x: x['time'])
                self._current_sequence = flattened_frames
                self._sequence_start_time = time.time()
                self._sequence_request = request if request_id else None
                self._publish_request_status(request, "accepted")
                self.get_logger().info(f"[Sequence] Playing sequence: {seq_name} ({len(flattened_frames)} frames)")
            else:
                self._explicit_motion_active = False
                self.get_logger().warn(f"[Sequence] Sequence '{seq_name}' not found or empty")
                self._publish_request_status(request, "rejected", "unknown_or_empty_sequence")

        elif tool == "stop_all":
            self._stop_motors(status="interrupted", detail="stop_all")
            self._publish_request_status(request, "completed")

    def _on_tracking_servo_targets(self, msg):
        """Update interpolated tracking targets without interrupting actions."""
        if self._game_active:
            return
        try:
            payload = json.loads(msg.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._apply_servo_targets(
            payload.get("targets", {}),
            payload.get("step_size", 40.0),
        )

    def _on_dialog_expression_targets(self, msg):
        """Apply low-priority dialogue targets without interrupting actions."""
        try:
            payload = json.loads(msg.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        pending = (payload.get("targets", {}), payload.get("step_size", 12.0))
        if self._game_active or self._explicit_motion_active or self._active_motor_cmd:
            self._pending_dialog_expression = pending
            return
        self._pending_dialog_expression = None
        self._apply_servo_targets(*pending)

    def _publish_request_status(self, request, status, detail=""):
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if not request_id:
            return
        self.action_status_pub.publish(String(data=build_action_status(
            request_id,
            request.get("name", "unknown"),
            status,
            source="sequence_ros_node",
            detail=detail,
        )))

    def _interrupt_sequence(self, detail):
        request = self._sequence_request
        self._sequence_request = None
        if request is not None:
            self._publish_request_status(request, "interrupted", detail)

    def _on_action_cancel(self, message):
        cancellation = parse_action_cancel(message.data)
        if cancellation is None:
            return
        request_id = cancellation["request_id"]
        reason = cancellation["reason"]
        if (
            self._sequence_request is not None
            and self._sequence_request.get("request_id") == request_id
        ):
            self._interrupt_sequence(reason)
            self._current_sequence = []
            self._explicit_motion_active = False
            for name in self._steps:
                self._steps[name] = 0.0
            if self._auto_reset_timer is not None:
                self.destroy_timer(self._auto_reset_timer)
                self._auto_reset_timer = None
            # Timeline motor frames have no separate motor request. They are
            # owned by this sequence and must stop along with its servo frames.
            if self._active_motor_cmd is not None and self._motor_request is None:
                self._stop_motors(status="interrupted", detail=reason)
        if (
            self._motor_request is not None
            and self._motor_request.get("request_id") == request_id
        ):
            self._stop_motors(status="interrupted", detail=reason)

    def _flatten_sequence(self, seq_name, offset_time=0.0, depth=0):
        """递归解析序列，将其扁平化为一维时间轴"""
        return self._sequence_library.flatten(
            seq_name,
            offset_time=offset_time,
            depth=depth,
        )

    def _reset_servos_to_init(self):
        self.get_logger().info("[Sequence] Auto-resetting servos to init state")
        self._trajectory.reset_to_initial(step_size=2.0)
        if self._auto_reset_timer:
            self.destroy_timer(self._auto_reset_timer)
            self._auto_reset_timer = None

    def _dispatch_action(self, act):
        t = act.get('type')
        if t == 'servo':
            name = act.get('name')
            if name in self._servos_config:
                # 兼容 angle 字段（如果有），但更推荐直接使用 pwm 字段
                val = act.get('pwm', act.get('angle', 4000))
                target_pwm = self._clamp_pwm(name, val)
                if target_pwm is not None:
                    self._targets[name] = target_pwm
                    self._steps[name] = float(act.get('step_size', 40.0))
                    
        elif t == 'pose':
            pose_name = act.get('name')
            pose_data = self._poses.get(pose_name)
            if pose_data:
                override_step = act.get('step_size')
                default_step = pose_data.get('default_step', 2.0)
                final_step = float(override_step if override_step is not None else default_step)
                
                for s_name, s_pwm in pose_data.get('targets', {}).items():
                    if s_name in self._servos_config:
                        target_pwm = self._clamp_pwm(s_name, s_pwm)
                        if target_pwm is not None:
                            self._targets[s_name] = target_pwm
                            self._steps[s_name] = final_step
                            
        elif t == 'motor':
            direction = act.get('direction', 'forward')
            duration = max(0.0, min(float(act.get('duration', 1.0)), 10.0))
            motor = self.MOTION_TO_MOTOR.get(direction)
            if motor:
                if duration <= 0.0:
                    self._stop_motors()
                    return
                self._active_motor_cmd = motor
                self._motor_stop_at = time.monotonic() + duration
                self._publish_active_motor()
                
        elif t == 'express_emotion':
            emotion = act.get('emotion', 'happy')
            msg = String()
            msg.data = f"eyeaction:{emotion}\n"
            self.tft_pub.publish(msg)
            
        elif t == 'manual_servo':
            self._apply_servo_targets(
                act.get('targets', {}),
                act.get('step_size', 30.0),
            )

    def _apply_servo_targets(self, targets, step_size):
        self._trajectory.apply_targets(targets, step_size)

    def _stop_motors(self, status="completed", detail=""):
        msg = String()
        msg.data = json.dumps(STOP_COMMAND, ensure_ascii=False)
        self.motor_pub.publish(msg)
        self._active_motor_cmd = None
        self._motor_stop_at = 0.0
        request = self._motor_request
        self._motor_request = None
        if request is not None:
            self._publish_request_status(request, status, detail)

    def _publish_active_motor(self):
        if self._active_motor_cmd is None:
            return
        self.motor_pub.publish(
            String(data=json.dumps(self._active_motor_cmd, ensure_ascii=False))
        )

    def _tick(self):
        if self._game_active:
            return
        if self._active_motor_cmd is not None:
            if time.monotonic() >= self._motor_stop_at:
                self._stop_motors()
            else:
                self._publish_active_motor()

        # 1. 时间轴播放器：按时间触发关键帧剧本
        if self._current_sequence:
            item = self._current_sequence[0]
            if time.time() - self._sequence_start_time >= item.get('time', 0):
                self._current_sequence.pop(0)
                for act in item.get('actions', []):
                    self._dispatch_action(act)

        # 2. The ROS-independent trajectory service applies mechanical
        # constraints and advances one 50 Hz interpolation step.
        changed_servos = self._trajectory.tick()

        # 3. Publish only positions changed by this control tick.
        for name, pwm in changed_servos.items():
            msg = String()
            msg.data = json.dumps({"name": name, "pwm": pwm})
            self.servo_pub.publish(msg)

        if (
            self._sequence_request is not None
            and not self._current_sequence
            and self._active_motor_cmd is None
            and all(
                self._steps.get(name, 0.0) <= 0.0
                or self._virtual_state.get(name) == self._targets.get(name)
                for name in self._virtual_state
            )
        ):
            request = self._sequence_request
            self._sequence_request = None
            self._publish_request_status(request, "completed")

        if (
            self._explicit_motion_active
            and not self._current_sequence
            and self._active_motor_cmd is None
            and all(
                self._steps.get(name, 0.0) <= 0.0
                or self._virtual_state.get(name) == self._targets.get(name)
                for name in self._virtual_state
            )
        ):
            self._explicit_motion_active = False
            if self._pending_dialog_expression is not None:
                pending = self._pending_dialog_expression
                self._pending_dialog_expression = None
                self._apply_servo_targets(*pending)

    def _on_game_state(self, message):
        active = game_is_active(message.data)
        if active and not self._game_active:
            self._interrupt_sequence("game_mode")
            self._current_sequence = []
            for name in self._steps:
                self._steps[name] = 0.0
            if self._auto_reset_timer:
                self.destroy_timer(self._auto_reset_timer)
                self._auto_reset_timer = None
            self._stop_motors(status="interrupted", detail="game_mode")
        self._game_active = active

def main(args=None):
    rclpy.init(args=args)
    node = SequenceRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
