"""PCA9685 15 通道纯输出状态与编码规则。"""

from __future__ import annotations
from typing import Mapping


class Pca9685OutputState:
    """维护 15 个 PCA9685 输出通道的纯状态类。

    0-8 通道：舵机通道。
    9-14 通道：电机通道（左电机 9,10,11；右电机 12,13,14）。
    """

    _DUTY_MIN: int = 1638
    _DUTY_MAX: int = 8192
    _MOTOR_HIGH: int = 65535
    _MOTOR_LOW: int = 0

    def __init__(self, servo_inits: Mapping[int, int] | None = None) -> None:
        default_val = int(self._DUTY_MIN + (self._DUTY_MAX - self._DUTY_MIN) * 90 / 180)
        self._channels: list[int] = [default_val] * 15

        if servo_inits:
            for ch, val in servo_inits.items():
                if 0 <= ch < 15:
                    self._channels[ch] = int(val)

        # 强制将电机通道 9–14 置零
        for ch in range(9, 15):
            self._channels[ch] = 0

        self._dirty: bool = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def set_pwm(self, channel: int, value: int) -> None:
        """设置指定通道的 PWM 原始值，仅在实际变化时标记 dirty。"""
        if not (0 <= channel < 15):
            return
        val = int(value)
        if self._channels[channel] != val:
            self._channels[channel] = val
            self._dirty = True

    def set_angle(self, channel: int, angle: float) -> None:
        """按 0~180° 公式设置指定通道角度（不 clamp），仅实际变化时标记 dirty。"""
        duty = int(self._DUTY_MIN + (self._DUTY_MAX - self._DUTY_MIN) * angle / 180)
        self.set_pwm(channel, duty)

    def set_motor(self, side: str, action: int, throttle: int) -> None:
        """设置电机通道。

        left 对应 9,10,11；right 对应 12,13,14。
        action=1 -> HIGH, LOW
        action=2 -> LOW, HIGH
        其他 -> LOW, LOW 且 throttle 强制为 0
        PWM 严格使用 int(throttle / 100.0 * 65535)。
        仅实际变化时标记 dirty。
        """
        if side == "left":
            base_ch = 9
        elif side == "right":
            base_ch = 12
        else:
            return

        if action == 1:
            in1 = self._MOTOR_HIGH
            in2 = self._MOTOR_LOW
        elif action == 2:
            in1 = self._MOTOR_LOW
            in2 = self._MOTOR_HIGH
        else:
            in1 = self._MOTOR_LOW
            in2 = self._MOTOR_LOW
            throttle = 0

        pwm = int(throttle / 100.0 * self._MOTOR_HIGH)

        in1_ch = base_ch
        in2_ch = base_ch + 1
        pwm_ch = base_ch + 2

        changed = False
        if self._channels[in1_ch] != in1:
            self._channels[in1_ch] = in1
            changed = True
        if self._channels[in2_ch] != in2:
            self._channels[in2_ch] = in2
            changed = True
        if self._channels[pwm_ch] != pwm:
            self._channels[pwm_ch] = pwm
            changed = True

        if changed:
            self._dirty = True

    def stop_motors(self) -> None:
        """停止两侧电机。"""
        self.set_motor("left", 0, 0)
        self.set_motor("right", 0, 0)

    def encode(self) -> str:
        """返回完整 payload 字符串：pca9685:v0,v1,...,v14。"""
        return "pca9685:" + ",".join(str(v) for v in self._channels)

    def mark_published(self) -> None:
        """清除 dirty 标记。"""
        self._dirty = False
