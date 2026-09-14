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
    DEFAULT_MOTION_TO_MOTOR,
    SequenceLibrary,
    SequenceRuntime,
    ServoTrajectory,
    load_yaml_mapping,
)

class SequenceRosNode(Node):
    # 所有的动作预设已迁移至 sequences.yaml，由 _flatten_sequence 处理

    MOTION_TO_MOTOR = DEFAULT_MOTION_TO_MOTOR

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
        self._runtime = SequenceRuntime(
            self._sequence_library,
            self._trajectory,
            motion_to_motor=self.MOTION_TO_MOTOR,
        )

        # Requests stay in the ROS adapter because their completion is emitted
        # through the action-status protocol.
        self._motor_request = None
        self._sequence_request = None
        self._auto_reset_timer = None
        self._game_active = False
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

    # Compatibility properties keep diagnostics and existing integration tests
    # stable while runtime state is now owned by the service.
    @property
    def _current_sequence(self):
        return self._runtime.timeline

    @_current_sequence.setter
    def _current_sequence(self, value):
        self._runtime.timeline = value

    @property
    def _sequence_start_time(self):
        return self._runtime.sequence_started_at

    @_sequence_start_time.setter
    def _sequence_start_time(self, value):
        self._runtime.sequence_started_at = value

    @property
    def _active_motor_cmd(self):
        return self._runtime.active_motor_command

    @_active_motor_cmd.setter
    def _active_motor_cmd(self, value):
        self._runtime.active_motor_command = value

    @property
    def _motor_stop_at(self):
        return self._runtime.motor_stop_at

    @_motor_stop_at.setter
    def _motor_stop_at(self, value):
        self._runtime.motor_stop_at = value

    @property
    def _explicit_motion_active(self):
        return self._runtime.explicit_motion_active

    @_explicit_motion_active.setter
    def _explicit_motion_active(self, value):
        self._runtime.explicit_motion_active = value
        
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
        self._runtime.clear_sequence()
        self._runtime.halt_interpolation()
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

            frame_count = self._runtime.start_sequence(seq_name, now=time.time())
            if frame_count:
                self._sequence_request = request if request_id else None
                self._publish_request_status(request, "accepted")
                self.get_logger().info(
                    f"[Sequence] Playing sequence: {seq_name} ({frame_count} frames)"
                )
            else:
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
            self._runtime.clear_sequence(clear_explicit_motion=True)
            self._runtime.halt_interpolation()
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
        effects = self._runtime.dispatch_action(
            act,
            monotonic_now=time.monotonic(),
        )
        self._publish_runtime_effects(effects)

    def _publish_runtime_effects(self, effects):
        for effect in effects:
            if effect.kind == "motor":
                self.motor_pub.publish(String(data=json.dumps(
                    effect.payload,
                    ensure_ascii=False,
                )))
            elif effect.kind == "motor_stop":
                self._stop_motors()
            elif effect.kind == "emotion":
                self.tft_pub.publish(String(data=f"eyeaction:{effect.payload}\n"))

    def _apply_servo_targets(self, targets, step_size):
        self._trajectory.apply_targets(targets, step_size)

    def _stop_motors(self, status="completed", detail=""):
        msg = String()
        msg.data = json.dumps(STOP_COMMAND, ensure_ascii=False)
        self.motor_pub.publish(msg)
        self._runtime.stop_motor()
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
        result = self._runtime.tick(
            wall_now=time.time(),
            monotonic_now=time.monotonic(),
        )
        self._publish_runtime_effects(result.effects)

        for name, pwm in result.servo_positions.items():
            msg = String()
            msg.data = json.dumps({"name": name, "pwm": pwm})
            self.servo_pub.publish(msg)

        if (
            self._sequence_request is not None
            and self._runtime.motion_complete()
        ):
            request = self._sequence_request
            self._sequence_request = None
            self._publish_request_status(request, "completed")

        if (
            self._explicit_motion_active
            and self._runtime.motion_complete()
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
            self._runtime.clear_sequence()
            self._runtime.halt_interpolation()
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
