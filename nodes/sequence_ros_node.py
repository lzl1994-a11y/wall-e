#!/usr/bin/env python3
# nodes/sequence_ros_node.py
# 统一轨迹控制器：接管所有 /action_cmd，支持单一动作与成组动作 (Timeline)，并利用步长进行平滑插值
import time
import json
import yaml
from services.action_command import parse_action_request
from services.action_status import ACTION_STATUS_TOPIC, build_action_status
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from services.motion_arbiter import MOTOR_AUTONOMY_TOPIC, STOP_COMMAND
from services.vision_pipeline_protocol import TRACKING_SERVO_TARGET_TOPIC
from services.dialog_expression_protocol import DIALOG_EXPRESSION_TARGET_TOPIC
from services.servo_motion_config import resolve_servo_target
from services.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active

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

        # 2. 初始化虚拟状态字典 (Virtual State)
        self._virtual_state = {}
        self._targets = {}
        self._steps = {}
        
        for name, cfg in self._servos_config.items():
            init_val = cfg.get('init', 150)
            self._virtual_state[name] = float(init_val)
            self._targets[name] = float(init_val)
            self._steps[name] = 0.0

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
        self.create_subscription(String, '/action_cmd', self._on_action_cmd, 10)
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
        
        self._first_tick = True
        self.get_logger().info('Sequence ROS Node online, taking over /action_cmd. 50Hz interpolation running.')
        
    def _load_yaml(self, path):
        import os
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Load {path} failed: {e}")
            return {}

    def _clamp_pwm(self, name, raw_pwm):
        """将传入的原始 PWM 值限制在安全的硬件限位内"""
        cfg = self._servos_config.get(name)
        return resolve_servo_target(cfg, raw_pwm) if cfg else None

    def _servo_init(self, name, fallback):
        cfg = self._servos_config.get(name, {})
        return float(cfg.get('init', fallback))

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

    def _flatten_sequence(self, seq_name, offset_time=0.0, depth=0):
        """递归解析序列，将其扁平化为一维时间轴"""
        if depth > 10:
            self.get_logger().error(f"Sequence max recursion depth exceeded at {seq_name}")
            return []
            
        frames = []
        seq = self._sequences.get(seq_name)
        if not seq:
            # 如果在 sequences 里没找到，但在 poses 里找到了，就临时包成一个单帧的动作
            if seq_name in self._poses:
                return [{'time': offset_time, 'actions': [{'type': 'pose', 'name': seq_name}]}]
            return frames
            
        # 兼容旧版本带有 loop_hz 字典的情况，如果是列表则直接遍历
        if isinstance(seq, dict):
            # 去除配置字段，只提取带 time 的列表项
            items = [v for k, v in seq.items() if isinstance(v, list)]
            if items:
                seq = items[0] # 提取包含 actions 的列表
            else:
                return []
                
        for item in seq:
            if not isinstance(item, dict) or 'time' not in item:
                continue
                
            t = item['time'] + offset_time
            actions = []
            
            for act in item.get('actions', []):
                if act.get('type') == 'sequence':
                    # 发现子序列，递归展开，并将子序列的起点加上当前的时间偏移
                    sub_frames = self._flatten_sequence(act.get('name'), offset_time=t, depth=depth+1)
                    frames.extend(sub_frames)
                else:
                    actions.append(act)
                    
            if actions:
                frames.append({'time': t, 'actions': actions})
                
        return frames

    def _reset_servos_to_init(self):
        self.get_logger().info("[Sequence] Auto-resetting servos to init state")
        for name, cfg in self._servos_config.items():
            self._targets[name] = float(cfg['init'])
            self._steps[name] = 2.0 # 默认柔和回中速度
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
        if not isinstance(targets, dict):
            return
        try:
            step_size = max(1.0, min(float(step_size), 1000.0))
        except (TypeError, ValueError):
            return
        for s_name, s_pwm in targets.items():
            if s_name in self._servos_config:
                target_pwm = self._clamp_pwm(s_name, s_pwm)
                if target_pwm is not None:
                    self._targets[s_name] = target_pwm
                    self._steps[s_name] = step_size

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

        # --- 2. 动态防碰撞：目标值修正 (Target Adjustments) ---
        # 头眼联动以 config.yaml 的 init 为分界：右转头限制左眼，左转头限制右眼。
        head_center = self._servo_init('head_yaw', 5000)
        eye_r_init = self._servo_init('eye_r', 3000)
        eye_l_init = self._servo_init('eye_l', 6500)
        eye_gap = 3000.0
        t_head = self._targets.get('head_yaw', head_center)

        # 规则1: 左转头时，右眼必须不低于右眼初始值，即 3000~4300。
        if t_head > head_center:
            if self._targets.get('eye_r', eye_r_init) < eye_r_init:
                self._targets['eye_r'] = eye_r_init
                if self._steps.get('eye_r', 0) <= 0: self._steps['eye_r'] = 30.0

        # 规则2: 右转头时，左眼必须不高于左眼初始值，即 6500~5000。
        if t_head < head_center:
            if self._targets.get('eye_l', eye_l_init) > eye_l_init:
                self._targets['eye_l'] = eye_l_init
                if self._steps.get('eye_l', 0) <= 0: self._steps['eye_l'] = 30.0

        # 规则3: 跷跷板联动机制 (eye_l - eye_r >= 3000)。
        # 右转头时不能为了满足跷跷板而把左眼推回 6500 以上，只能压低右眼。
        t_eye_r = self._targets.get('eye_r', eye_r_init)
        t_eye_l = self._targets.get('eye_l', eye_l_init)

        if t_head < head_center:
            max_r = t_eye_l - eye_gap
            if t_eye_r > max_r:
                self._targets['eye_r'] = max_r
                if self._steps.get('eye_r', 0) <= 0: self._steps['eye_r'] = 30.0
        else:
            min_l = t_eye_r + eye_gap
            if t_eye_l < min_l:
                self._targets['eye_l'] = min_l
                if self._steps.get('eye_l', 0) <= 0: self._steps['eye_l'] = 30.0

            t_eye_l = self._targets.get('eye_l', eye_l_init)
            max_r = t_eye_l - eye_gap
            if t_eye_r > max_r:
                self._targets['eye_r'] = max_r
                if self._steps.get('eye_r', 0) <= 0: self._steps['eye_r'] = 30.0

        # --- 3. 轨迹控制器：50Hz 舵机高频插值与瞬态限位 ---
        changed_servos = set()
        
        if self._first_tick:
            self._first_tick = False
            for name in self._virtual_state:
                changed_servos.add(name)
                
        for name in list(self._virtual_state.keys()):
            target = self._targets[name]
            step = self._steps[name]
            current = self._virtual_state[name]
            
            if step <= 0 or current == target:
                continue
                
            next_val = current
            if abs(target - current) <= step:
                next_val = target
            elif target > current:
                next_val += step
            else:
                next_val -= step
                
            # 瞬态拦截：防止在走向安全目标的过程中，发生中间态物理干涉
            if name == 'head_yaw' and next_val > head_center:
                if self._virtual_state.get('eye_r', eye_r_init) < eye_r_init:
                    next_val = head_center  # 右眼还没到初始值以上，不许头往左转

            if name == 'eye_r' and next_val < eye_r_init:
                if self._virtual_state.get('head_yaw', head_center) > head_center:
                    next_val = eye_r_init  # 头还在左边，右眼不许低于初始值

            if name == 'head_yaw' and next_val < head_center:
                if self._virtual_state.get('eye_l', eye_l_init) > eye_l_init:
                    next_val = head_center  # 左眼还没到初始值以下，不许头往右转

            if name == 'eye_l' and next_val > eye_l_init:
                if self._virtual_state.get('head_yaw', head_center) < head_center:
                    next_val = eye_l_init  # 头还在右边，左眼不许高于初始值
                    
            # 瞬态拦截：跷跷板联动 (eye_l - eye_r >= 3000)
            if name == 'eye_l':
                v_eye_r = self._virtual_state.get('eye_r', eye_r_init)
                min_allow = v_eye_r + eye_gap
                if self._virtual_state.get('head_yaw', head_center) < head_center:
                    next_val = min(next_val, eye_l_init)
                elif next_val < min_allow:
                    next_val = min_allow
                    
            if name == 'eye_r':
                v_eye_l = self._virtual_state.get('eye_l', eye_l_init)
                max_allow = v_eye_l - eye_gap
                if next_val > max_allow:
                    next_val = max_allow
                    
            self._virtual_state[name] = next_val
            changed_servos.add(name)
            
        # 4. 发布状态
        for name in changed_servos:
            msg = String()
            msg.data = json.dumps({"name": name, "pwm": int(self._virtual_state[name])})
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
