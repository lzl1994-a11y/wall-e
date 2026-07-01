#!/usr/bin/env python3
# nodes/sequence_ros_node.py
# 统一轨迹控制器：接管所有 /action_cmd，支持单一动作与成组动作 (Timeline)，并利用步长进行平滑插值
import json
import time
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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
        self._motor_timer = None
        self._auto_reset_timer = None

        # 3. ROS 接口
        self.servo_pub = self.create_publisher(String, '/servo_cmd', 10)
        self.motor_pub = self.create_publisher(String, '/motor_cmd', 10)
        self.tft_pub   = self.create_publisher(String, '/tft_cmd', 10)

        # 4. 核心 50Hz 插值定时器
        self.create_timer(0.02, self._tick)

        # 统一订阅 /action_cmd (接管之前 action_ros_node 的职责)
        self.create_subscription(String, '/action_cmd', self._on_action_cmd, 10)
        
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
        if not cfg: return None
        l1 = cfg['limit_1']
        l2 = cfg['limit_2']
        min_pwm = min(l1, l2)
        max_pwm = max(l1, l2)
        return max(min_pwm, min(max_pwm, raw_pwm))

    def _on_action_cmd(self, msg):
        try:
            cmd = json.loads(msg.data)
        except Exception:
            return
            
        tool = cmd.get("name")
        args = cmd.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}

        # ===== 外部打断机制核心：清空队列，并清零步长 =====
        self._current_sequence = [] # 打断成组动作
        for name in self._steps:
            self._steps[name] = 0.0 # 清零步长，平滑运动瞬间停止
        if self._auto_reset_timer:
            self.destroy_timer(self._auto_reset_timer)
            self._auto_reset_timer = None
        self.get_logger().info(f"[Interrupt] Cleared state for tool: {tool}")

        # ===== 指令分发 =====
        if tool == "express_emotion":
            self._dispatch_action({"type": "express_emotion", "emotion": args.get("emotion", "happy")})
            
        elif tool == "move_chassis":
            self._dispatch_action({
                "type": "motor", 
                "direction": args.get("direction", "forward"), 
                "duration": float(args.get("duration", 1.0))
            })
            
        elif tool == "manual_servo":
            self._dispatch_action({
                "type": "manual_servo",
                "targets": args.get("targets", {}),
                "step_size": args.get("step_size", 30.0)
            })
            

        elif tool == "play_sequence":
            seq_name = args.get("sequence_name", "")
            
            # 使用时间轴扁平化算法拆解嵌套序列
            flattened_frames = self._flatten_sequence(seq_name, offset_time=0.0)
            if flattened_frames:
                # 按照绝对时间进行排序
                flattened_frames.sort(key=lambda x: x['time'])
                self._current_sequence = flattened_frames
                self._sequence_start_time = time.time()
                self.get_logger().info(f"[Sequence] Playing sequence: {seq_name} ({len(flattened_frames)} frames)")
            else:
                self.get_logger().warn(f"[Sequence] Sequence '{seq_name}' not found or empty")

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
            duration = act.get('duration', 1.0)
            motor = self.MOTION_TO_MOTOR.get(direction)
            if motor:
                msg = String()
                msg.data = json.dumps(motor, ensure_ascii=False)
                self.motor_pub.publish(msg)
                if self._motor_timer:
                    self.destroy_timer(self._motor_timer)
                self._motor_timer = self.create_timer(duration, self._stop_motors)
                
        elif t == 'express_emotion':
            emotion = act.get('emotion', 'happy')
            msg = String()
            msg.data = f"eyeaction:{emotion}\n"
            self.tft_pub.publish(msg)
            
        elif t == 'manual_servo':
            targets = act.get('targets', {})
            step_size = float(act.get('step_size', 30.0))
            for s_name, s_pwm in targets.items():
                if s_name in self._servos_config:
                    target_pwm = self._clamp_pwm(s_name, s_pwm)
                    if target_pwm is not None:
                        self._targets[s_name] = target_pwm
                        self._steps[s_name] = step_size

    def _stop_motors(self):
        msg = String()
        msg.data = json.dumps({"left": {"action": 0, "throttle": 0}, "right": {"action": 0, "throttle": 0}}, ensure_ascii=False)
        self.motor_pub.publish(msg)
        if self._motor_timer:
            self.destroy_timer(self._motor_timer)
            self._motor_timer = None

    def _tick(self):
        # 1. 时间轴播放器：按时间触发关键帧剧本
        if self._current_sequence:
            item = self._current_sequence[0]
            if time.time() - self._sequence_start_time >= item.get('time', 0):
                self._current_sequence.pop(0)
                for act in item.get('actions', []):
                    self._dispatch_action(act)

        # --- 2. 动态防碰撞：目标值修正 (Target Adjustments) ---
        # 确保系统最终收敛到一个安全的静态目标
        t_head = self._targets.get('head_yaw', 5000)
        
        # 规则1: 左转头时如果右眼低于3000则设置为3000 (头左偏是 > 5000)
        if t_head > 5000:
            if self._targets.get('eye_r', 3000) < 3000:
                self._targets['eye_r'] = 3000
                if self._steps.get('eye_r', 0) <= 0: self._steps['eye_r'] = 30.0
                
        # 规则2: 右转头情况下左眼不能大于6000 (头右偏是 < 5000)
        if t_head < 5000:
            if self._targets.get('eye_l', 6000) > 6000:
                self._targets['eye_l'] = 6000
                if self._steps.get('eye_l', 0) <= 0: self._steps['eye_l'] = 30.0
                
        # 规则3: 跷跷板联动机制 (抽象数学约束: eye_l - eye_r >= 3000)
        # 防止左眼和右眼同时过度抬起导致的机械干涉
        t_eye_r = self._targets.get('eye_r', 3000)
        t_eye_l = self._targets.get('eye_l', 6000)
        
        min_l = t_eye_r + 3000
        if t_eye_l < min_l:
            self._targets['eye_l'] = min_l
            if self._steps.get('eye_l', 0) <= 0: self._steps['eye_l'] = 30.0
            
        t_eye_l = self._targets.get('eye_l', 6000)
        max_r = t_eye_l - 3000
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
            if name == 'head_yaw' and next_val > 5000:
                if self._virtual_state.get('eye_r', 3000) < 3000:
                    next_val = 5000  # 眼睛还没抬起来，不许头往左转
                    
            if name == 'eye_r' and next_val < 3000:
                if self._virtual_state.get('head_yaw', 5000) > 5000:
                    next_val = 3000  # 头还在左边没回正，不许眼睛往下低
                    
            if name == 'head_yaw' and next_val < 5000:
                if self._virtual_state.get('eye_l', 6000) > 6000:
                    next_val = 5000  # 眼睛还没抬起来，不许头往右转
                    
            if name == 'eye_l' and next_val > 6000:
                if self._virtual_state.get('head_yaw', 5000) < 5000:
                    next_val = 6000  # 头还在右边没回正，不许眼睛往下低
                    
            # 瞬态拦截：跷跷板联动 (eye_l - eye_r >= 3000)
            if name == 'eye_l':
                v_eye_r = self._virtual_state.get('eye_r', 3000)
                min_allow = v_eye_r + 3000
                if next_val < min_allow:
                    next_val = min_allow
                    
            if name == 'eye_r':
                v_eye_l = self._virtual_state.get('eye_l', 6000)
                max_allow = v_eye_l - 3000
                if next_val > max_allow:
                    next_val = max_allow
                    
            self._virtual_state[name] = next_val
            changed_servos.add(name)
            
        # 4. 发布状态
        for name in changed_servos:
            msg = String()
            msg.data = json.dumps({"name": name, "pwm": int(self._virtual_state[name])})
            self.servo_pub.publish(msg)

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
