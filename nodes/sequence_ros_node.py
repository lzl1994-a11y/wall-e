#!/usr/bin/env python3
# nodes/sequence_ros_node.py
# 统一轨迹控制器：接收仲裁后的 /action_cmd，支持单一动作与成组动作 (Timeline)
import time
import json
from services.action.action_cancel import ACTION_CANCEL_TOPIC, parse_action_cancel
from services.action.action_command import ACTION_COMMAND_TOPIC, parse_action_request
from services.action.action_status import ACTION_STATUS_TOPIC, build_action_status
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from services.motion.motion_arbiter import MOTOR_AUTONOMY_TOPIC, STOP_COMMAND
from services.vision.vision_pipeline_protocol import TRACKING_SERVO_TARGET_TOPIC
from services.dialog.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TOPIC,
    DIALOG_EXPRESSION_TARGET_TOPIC,
    encode_dialog_expression,
)
from services.game.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active
from services.motion.sequence_execution import (
    DEFAULT_MOTION_TO_MOTOR,
    SequenceCommandController,
    SequenceLibrary,
    SequenceRuntime,
    ServoTrajectory,
    load_yaml_mapping,
)
from services.motion.choreography import ChoreographyStore
from services.audio.music_protocol import (
    CHOREOGRAPHY_MUSIC_SYNC_TOPIC,
    decode_choreography_music_sync,
)

class SequenceRosNode(Node):
    # 所有动作预设由 SequenceLibrary 从 sequences.yaml 解析。

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
        self._choreography_store = ChoreographyStore('core/config.yaml')
        self._controller = SequenceCommandController(
            self._runtime,
            choreography_loader=self._load_choreography_playback,
        )

        self._auto_reset_timer = None
        self._pending_choreography = None
        self._early_music_sync = {}

        # 3. ROS 接口
        self.servo_pub = self.create_publisher(String, '/servo_cmd', 10)
        self.motor_pub = self.create_publisher(String, MOTOR_AUTONOMY_TOPIC, 10)
        self.tft_pub   = self.create_publisher(String, '/tft_cmd', 10)
        self.dialog_expression_pub = self.create_publisher(
            String, DIALOG_EXPRESSION_TOPIC, 10
        )
        self.action_status_pub = self.create_publisher(String, ACTION_STATUS_TOPIC, 10)

        # 4. 核心 50Hz 插值定时器
        self.create_timer(0.02, self._tick)

        # 统一订阅 /action_cmd，负责动作编排和运动指令分发
        self.create_subscription(String, ACTION_COMMAND_TOPIC, self._on_action_cmd, 10)
        self.create_subscription(String, ACTION_CANCEL_TOPIC, self._on_action_cancel, 10)
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(
            String, CHOREOGRAPHY_MUSIC_SYNC_TOPIC, self._on_music_sync, 10
        )
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

    @property
    def _motor_request(self):
        return self._controller.motor_request

    @_motor_request.setter
    def _motor_request(self, value):
        self._controller.motor_request = value

    @property
    def _sequence_request(self):
        return self._controller.sequence_request

    @_sequence_request.setter
    def _sequence_request(self, value):
        self._controller.sequence_request = value

    @property
    def _pending_dialog_expression(self):
        return self._controller.pending_dialog_expression

    @_pending_dialog_expression.setter
    def _pending_dialog_expression(self, value):
        self._controller.pending_dialog_expression = value

    @property
    def _game_active(self):
        return self._controller.game_active

    @_game_active.setter
    def _game_active(self, value):
        self._controller.game_active = value
        
    def _load_yaml(self, path):
        return load_yaml_mapping(path, on_error=self.get_logger().error)

    def _load_choreography_playback(self, choreography_id):
        document = self._choreography_store.get(choreography_id)
        _, compiled = self._choreography_store.validate(document)
        return {
            **compiled.get("playback", {}),
            "has_audio": bool((compiled.get("audio") or {}).get("asset_id")),
        }

    def _on_music_sync(self, message):
        event = decode_choreography_music_sync(message.data)
        if event is None:
            return
        pending = self._pending_choreography
        if pending is not None and event["playback_id"] == pending["playback_id"]:
            self._apply_music_sync(event["ready"])
        else:
            self._early_music_sync[event["playback_id"]] = (event["ready"], time.monotonic())

    def _apply_music_sync(self, ready):
        if ready:
            self._start_pending_choreography()
        else:
            self.get_logger().error("编排音乐未能输出音频，已取消动作")
            self._publish_request_status(
                self._pending_choreography["request"], "failed", "music_start_failed"
            )
            self._pending_choreography = None

    def _clear_pending_choreography(self, detail, *, report_status=True):
        pending = self._pending_choreography
        self._pending_choreography = None
        if pending is not None and report_status:
            self._publish_request_status(pending["request"], "interrupted", detail)

    def _start_pending_choreography(self):
        pending = self._pending_choreography
        if pending is None:
            return
        self._pending_choreography = None
        frame_count = self._runtime.start_timeline(
            pending["frames"],
            now=time.time(),
            preview=False,
            motor_duration_limit=30.0,
        )
        self.get_logger().info(
            f"[Choreography] Timeline started "
            f"({pending['choreography_id']}, {frame_count} frames)"
        )
        self._publish_request_status(
            pending["request"],
            "completed" if frame_count else "failed",
            "playback_started" if frame_count else "empty_choreography",
        )

    def _on_action_cmd(self, msg):
        request = parse_action_request(msg.data)
        if request is None:
            return
        if request.get("name") in self._controller.SUPPORTED_ACTIONS:
            self._clear_pending_choreography("superseded_by_new_command")
        effects = self._controller.handle_action(
            request,
            wall_now=time.time(),
            monotonic_now=time.monotonic(),
            defer_choreography=request.get("name") == "play_choreography",
        )
        self._publish_runtime_effects(effects)

    def _on_tracking_servo_targets(self, msg):
        """Update interpolated tracking targets without interrupting actions."""
        try:
            payload = json.loads(msg.data)
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        self._controller.apply_tracking_targets(
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
        self._controller.apply_dialog_expression(
            payload.get("targets", {}),
            payload.get("step_size", 12.0),
        )

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

    def _on_action_cancel(self, message):
        cancellation = parse_action_cancel(message.data)
        if cancellation is None:
            return
        pending = self._pending_choreography
        if pending is not None and pending["request"].get("request_id") == cancellation.get("request_id"):
            self._clear_pending_choreography("cancelled", report_status=False)
        self._publish_runtime_effects(self._controller.cancel(cancellation))

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
                self.dialog_expression_pub.publish(String(
                    data=encode_dialog_expression(effect.payload, "medium")
                ))
            elif effect.kind == "status":
                self._publish_request_status(
                    effect.payload["request"],
                    effect.payload["status"],
                    effect.payload["detail"],
                )
            elif effect.kind == "cancel_auto_reset_timer":
                if self._auto_reset_timer is not None:
                    self.destroy_timer(self._auto_reset_timer)
                    self._auto_reset_timer = None
            elif effect.kind == "log_info":
                self.get_logger().info(effect.payload)
            elif effect.kind == "log_warning":
                self.get_logger().warn(effect.payload)
            elif effect.kind == "queue_choreography":
                self._pending_choreography = effect.payload
                cutoff = time.monotonic() - 5.0
                self._early_music_sync = {
                    key: event for key, event in self._early_music_sync.items()
                    if event[1] >= cutoff
                }
                if not effect.payload["has_audio"]:
                    self._start_pending_choreography()
                else:
                    early = self._early_music_sync.pop(effect.payload["playback_id"], None)
                    if early is not None:
                        self._apply_music_sync(early[0])

    def _stop_motors(self, status="completed", detail=""):
        msg = String()
        msg.data = json.dumps(STOP_COMMAND, ensure_ascii=False)
        self.motor_pub.publish(msg)
        self._runtime.stop_motor()
        request = self._motor_request
        self._motor_request = None
        if request is not None:
            self._publish_request_status(request, status, detail)

    def _tick(self):
        if self._game_active:
            return
        result = self._controller.tick(
            wall_now=time.time(),
            monotonic_now=time.monotonic(),
        )
        self._publish_runtime_effects(result.effects)

        for name, pwm in result.servo_positions.items():
            msg = String()
            msg.data = json.dumps({"name": name, "pwm": pwm})
            self.servo_pub.publish(msg)

    def _on_game_state(self, message):
        active = game_is_active(message.data)
        if active:
            self._clear_pending_choreography("game_mode_active")
            self._early_music_sync.clear()
        self._publish_runtime_effects(self._controller.set_game_active(active))

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
