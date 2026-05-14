# services/servo_control.py
import time
import threading
import Hobot.GPIO as GPIO
import board
import busio
from adafruit_pca9685 import PCA9685

class ServoControl:
    
    # ==========================================
    # 静态配置区 (Static Configurations)
    # ==========================================
    # 电机方向控制引脚 (RDK X3 GPIO BOARD 编码)
    L_IN1, L_IN2 = 11, 12  # 左轮方向
    R_IN1, R_IN2 = 13, 15  # 右轮方向
    
    # PCA9685 PWM 通道分配
    PWM_CH_L = 14          # 履带左轮速度
    PWM_CH_R = 15          # 履带右轮速度
    
    # 舵机通道映射表
    SERVO_MAP = {
        "eyebrow_r": 0, "eyebrow_l": 1, "eye_r": 2, "eye_l": 3,
        "head_yaw": 4, "neck_top": 5, "neck_bottom": 6, "arm_r": 7, "arm_l": 8
    }

    # PWM 脉宽常数 (针对 50Hz 频率下的 16bit 占空比换算)
    # 舵机通常 0度对应 0.5ms，180度对应 2.5ms
    # 20ms周期下：0.5ms 占空比约 1638，2.5ms 占空比约 8192
    SERVO_MIN_DUTY = 1638 
    SERVO_MAX_DUTY = 8192 

    def __init__(self, update_rate=10):
        print(f"[硬件初始化] ⚙️ ServoControl 底层驱动已启动 (频率: {update_rate}Hz)")
        self.update_rate = update_rate
        self._running = True

        # ==========================================
        # 1. 真实初始化 RDK X3 GPIO
        # ==========================================
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        for pin in [self.L_IN1, self.L_IN2, self.R_IN1, self.R_IN2]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # ==========================================
        # 2. 真实初始化 PCA9685
        # ==========================================
        print("   -> 正在连接 I2C 总线并唤醒 PCA9685...")
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50  # 舵机标准工作频率 50Hz

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
        print("   -> 硬件驱动加载完毕！")

    # ==========================================
    # 数学转换辅助函数
    # ==========================================
    def _angle_to_duty(self, angle):
        """将 0-180 度的角度转换为 PCA9685 的 16-bit 占空比数值"""
        # 线性插值映射
        duty = self.SERVO_MIN_DUTY + (angle / 180.0) * (self.SERVO_MAX_DUTY - self.SERVO_MIN_DUTY)
        return int(duty)

    def _throttle_to_duty(self, throttle):
        """将 0-100 的油门百分比转换为 PCA9685 的 16-bit 占空比数值"""
        # 0% -> 0, 100% -> 65535
        duty = (throttle / 100.0) * 65535
        return int(duty)

    # ==========================================
    # 外部调用 API
    # ==========================================
    def set_angle(self, name, angle):
        """设置舵机目标角度"""
        if name in self.target_angles:
            self.target_angles[name] = max(0, min(180, angle))

    def set_motor(self, side, action, throttle):
        """设置电机动作和油门 (side: 'track_l' 或 'track_r')"""
        if side in self.motors:
            self.motors[side]["action"] = action
            self.motors[side]["throttle"] = max(0, min(100, throttle))

    # ==========================================
    # 底层控制死循环
    # ==========================================
    def _control_loop(self):
        while self._running:
            start_t = time.time()

            # --- 1. 刷新 9 路舵机 ---
            for name, angle in self.target_angles.items():
                channel = self.SERVO_MAP[name]
                duty_cycle = self._angle_to_duty(angle)
                self.pca.channels[channel].duty_cycle = duty_cycle

            # --- 2. 刷新电机逻辑 (GPIO 挂挡 + PCA9685 踩油门) ---
            for side in ["track_l", "track_r"]:
                m = self.motors[side]
                act, thr = m["action"], m["throttle"]
                
                # 确定当前处理的是左轮还是右轮的引脚
                in1, in2 = (self.L_IN1, self.L_IN2) if side == "track_l" else (self.R_IN1, self.R_IN2)
                pwm_ch = self.PWM_CH_L if side == "track_l" else self.PWM_CH_R

                # GPIO 挂挡逻辑
                if act == 1:   # 正转 (前进)
                    GPIO.output(in1, GPIO.HIGH); GPIO.output(in2, GPIO.LOW)
                elif act == 2: # 反转 (后退)
                    GPIO.output(in1, GPIO.LOW);  GPIO.output(in2, GPIO.HIGH)
                else:          # 停止
                    GPIO.output(in1, GPIO.LOW);  GPIO.output(in2, GPIO.LOW)
                    thr = 0    # 强制油门归零，确保安全

                # PCA9685 输出 PWM 速度
                motor_duty = self._throttle_to_duty(thr)
                self.pca.channels[pwm_ch].duty_cycle = motor_duty

            # 精准维持指定的 Hz 频率
            elapsed = time.time() - start_t
            sleep_time = max(0, (1.0 / self.update_rate) - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        """系统退出时的安全清理操作"""
        self._running = False
        # 1. 舵机归中 (90度)
        for name in self.target_angles:
            self.target_angles[name] = 90
        # 2. 停止所有 PWM 和 GPIO
        time.sleep(0.2) # 等待最后一帧发送完毕
        self.pca.deinit()
        GPIO.cleanup()
        print("[硬件清理] ⚙️ 硬件连接已安全断开。")