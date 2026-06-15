# services/servo_control.py
# 瓦力底层硬件驱动：PCA9685 统一管理舵机(0-8)与 TB6612FNG 电机(9-14)
# 全部通过 I2C 一条总线控制，不再使用 GPIO 直连电机。
import time
import threading
import busio
from adafruit_pca9685 import PCA9685

class ServoControl:
    
    # ==========================================
    # 静态配置区 (Static Configurations)
    # ==========================================

    # ── PCA9685 通道分配 ──
    # 舵机: 通道 0-8（共9路）
    #   eyebrow_r=0, eyebrow_l=1, eye_r=2, eye_l=3,
    #   head_yaw=4, neck_top=5, neck_bottom=6, arm_r=7, arm_l=8
    # 电机(TB6612FNG): 通道 9-14（每电机3线：IN1/IN2/PWM）
    #   左电机: IN1=9, IN2=10, PWM=11
    #   右电机: IN1=12, IN2=13, PWM=14

    SERVO_MAP = {
        "eyebrow_r": 0, "eyebrow_l": 1, "eye_r": 2, "eye_l": 3,
        "head_yaw": 4, "neck_top": 5, "neck_bottom": 6, "arm_r": 7, "arm_l": 8
    }

    # TB6612FNG 电机通道
    CH_L_IN1, CH_L_IN2, CH_L_PWM = 9, 10, 11    # 左电机
    CH_R_IN1, CH_R_IN2, CH_R_PWM = 12, 13, 14  # 右电机

    # PWM 脉宽常数 (针对 50Hz 频率下的 16bit 占空比换算)
    # 舵机通常 0度对应 0.5ms，180度对应 2.5ms
    # 20ms周期下：0.5ms 占空比约 1638，2.5ms 占空比约 8192
    SERVO_MIN_DUTY = 1638 
    SERVO_MAX_DUTY = 8192 

    # 电机全高/全低值（16-bit）
    MOTOR_HIGH = 65535   # 3.3V 持续高电平 -> TB6612 IN=1
    MOTOR_LOW  = 0        # 0V 持续低电平   -> TB6612 IN=0

    def __init__(self, update_rate=10):
        print(f"[硬件初始化] ServoControl 底层驱动已启动 (频率: {update_rate}Hz)")
        self.update_rate = update_rate
        self._running = True

        # ==========================================
        # 初始化 PCA9685（I2C）
        # ==========================================
        print("   -> 正在连接 I2C 总线并唤醒 PCA9685...")
        import busio
        from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as _I2C
        self.i2c = busio.I2C(1)  # 强制指定 /dev/i2c-1
        self.i2c._i2c = _I2C(1)
        self.pca = PCA9685(self.i2c, address=0x70)
        self.pca.frequency = 50  # 舵机标准工作频率 50Hz，电机也兼容

        # 初始化电机通道为低电平（停止状态），防止上电瞬间电机抖动
        for ch in [self.CH_L_IN1, self.CH_L_IN2, self.CH_L_PWM,
                   self.CH_R_IN1, self.CH_R_IN2, self.CH_R_PWM]:
            self.pca.channels[ch].duty_cycle = 0

        # ==========================================
        # 状态黑板池 (State Blackboard)
        # ==========================================
        self.target_angles = {name: 90 for name in self.SERVO_MAP.keys()}
        self.motors = {
            "track_l": {"action": 0, "throttle": 0},
            "track_r": {"action": 0, "throttle": 0}
        }

        # 启动 10Hz 硬件刷新守护线程
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("   -> 硬件驱动加载完毕！（舵机0-8, 电机9-14, 全部走PCA9685）")

    # ==========================================
    # 数学转换辅助函数
    # ==========================================
    def _angle_to_duty(self, angle):
        """将 0-180 度的角度转换为 PCA9685 的 16-bit 占空比数值"""
        duty = self.SERVO_MIN_DUTY + (angle / 180.0) * (self.SERVO_MAX_DUTY - self.SERVO_MIN_DUTY)
        return int(duty)

    def _throttle_to_duty(self, throttle):
        """将 0-100 的油门百分比转换为 PCA9685 的 16-bit 占空比数值"""
        return int((throttle / 100.0) * 65535)

    # ==========================================
    # 外部调用 API
    # ==========================================
    def set_angle(self, name, angle):
        """设置舵机目标角度 (0-180)，中立位 90"""
        if name in self.target_angles:
            self.target_angles[name] = max(0, min(180, angle))

    def set_motor(self, side, action, throttle):
        """
        设置电机动作和油门
        side: 'track_l' 或 'track_r'
        action: 0=停止, 1=正转(前进), 2=反转(后退)
        throttle: 0-100 油门百分比
        """
        if side in self.motors:
            self.motors[side]["action"] = action
            self.motors[side]["throttle"] = max(0, min(100, throttle))

    # ==========================================
    # 底层控制死循环 (10Hz)
    # ==========================================
    def _control_loop(self):
        while self._running:
            start_t = time.time()

            # --- 1. 刷新 9 路舵机 (通道 0-8) ---
            for name, angle in self.target_angles.items():
                channel = self.SERVO_MAP[name]
                duty_cycle = self._angle_to_duty(angle)
                self.pca.channels[channel].duty_cycle = duty_cycle

            # --- 2. 刷新 2 路电机 (通道 9-14, TB6612FNG) ---
            # 左电机
            self._write_motor(
                self.CH_L_IN1, self.CH_L_IN2, self.CH_L_PWM,
                self.motors["track_l"]["action"],
                self.motors["track_l"]["throttle"]
            )
            # 右电机
            self._write_motor(
                self.CH_R_IN1, self.CH_R_IN2, self.CH_R_PWM,
                self.motors["track_r"]["action"],
                self.motors["track_r"]["throttle"]
            )

            # 精准维持指定的 Hz 频率
            elapsed = time.time() - start_t
            sleep_time = max(0, (1.0 / self.update_rate) - elapsed)
            time.sleep(sleep_time)

    def _write_motor(self, in1_ch, in2_ch, pwm_ch, action, throttle):
        """
        通过 PCA9685 三个通道控制一个 TB6612FNG 电机。
        IN1/IN2 用全高/全低模拟数字信号，PWM 通道用占空比调速。

        TB6612FNG 真值表:
          停止: IN1=LOW,  IN2=LOW
          正转: IN1=HIGH, IN2=LOW
          反转: IN1=LOW,  IN2=HIGH
        """
        if action == 1:   # 正转
            self.pca.channels[in1_ch].duty_cycle = self.MOTOR_HIGH
            self.pca.channels[in2_ch].duty_cycle = self.MOTOR_LOW
        elif action == 2: # 反转
            self.pca.channels[in1_ch].duty_cycle = self.MOTOR_LOW
            self.pca.channels[in2_ch].duty_cycle = self.MOTOR_HIGH
        else:             # 停止
            self.pca.channels[in1_ch].duty_cycle = self.MOTOR_LOW
            self.pca.channels[in2_ch].duty_cycle = self.MOTOR_LOW
            throttle = 0  # 强制油门归零

        self.pca.channels[pwm_ch].duty_cycle = self._throttle_to_duty(throttle)

    def stop(self):
        """系统退出时的安全清理操作"""
        self._running = False
        # 1. 舵机归中 (90度)
        for name in self.target_angles:
            self.target_angles[name] = 90
        # 2. 停止所有电机通道
        for ch in range(9, 15):
            self.pca.channels[ch].duty_cycle = 0
        time.sleep(0.2)
        self.pca.deinit()
        print("[硬件清理] 硬件连接已安全断开。")