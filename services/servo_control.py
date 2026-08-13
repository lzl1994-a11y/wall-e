"""Direct Ubuntu I2C driver for the PCA9685 servo and motor controller."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from services.motor_control import apply_direction_inversion, motor_inversion_flags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "core" / "config.yaml"


class _SMBusI2C:
    """Expose smbus2 as the busio-compatible API used by Adafruit drivers."""

    def __init__(self, bus_num: int):
        try:
            from smbus2 import SMBus, i2c_msg
        except ImportError as exc:
            raise RuntimeError("Ubuntu I2C 模式需要安装 smbus2") from exc

        self._bus = SMBus(bus_num)
        self._i2c_msg = i2c_msg
        self._lock = threading.Lock()
        self._closed = False

    def try_lock(self) -> bool:
        return self._lock.acquire(blocking=False)

    def unlock(self) -> None:
        if self._lock.locked():
            self._lock.release()

    def writeto(self, address: int, buffer: bytes, **kwargs: Any) -> None:
        start = int(kwargs.get("start", 0))
        end = int(kwargs.get("end", len(buffer)))
        payload = bytes(buffer[start:end])
        if not payload:
            self._bus.write_quick(address)
            return
        message = self._i2c_msg.write(address, payload)
        self._bus.i2c_rdwr(message)

    def readfrom_into(self, address: int, buffer: bytearray, **kwargs: Any) -> None:
        start = int(kwargs.get("start", 0))
        end = int(kwargs.get("end", len(buffer)))
        length = end - start
        if length <= 0:
            return
        message = self._i2c_msg.read(address, length)
        self._bus.i2c_rdwr(message)
        buffer[start:end] = bytes(message)

    def writeto_then_readfrom(
        self,
        address: int,
        out_buffer: bytes,
        in_buffer: bytearray,
        **kwargs: Any,
    ) -> None:
        out_start = int(kwargs.get("out_start", 0))
        out_end = int(kwargs.get("out_end", len(out_buffer)))
        in_start = int(kwargs.get("in_start", 0))
        in_end = int(kwargs.get("in_end", len(in_buffer)))
        output = bytes(out_buffer[out_start:out_end])
        input_length = in_end - in_start
        if input_length <= 0:
            return

        read_message = self._i2c_msg.read(address, input_length)
        if output:
            write_message = self._i2c_msg.write(address, output)
            self._bus.i2c_rdwr(write_message, read_message)
        else:
            self._bus.i2c_rdwr(read_message)
        in_buffer[in_start:in_end] = bytes(read_message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bus.close()


class ServoControl:
    """Own one PCA9685 instance and drive both servos and TB6612 channels."""

    SERVO_MIN_DUTY = 1638
    SERVO_MAX_DUTY = 8192
    MOTOR_HIGH = 65535
    MOTOR_LOW = 0

    CH_L_IN1, CH_L_IN2, CH_L_PWM = 9, 10, 11
    CH_R_IN1, CH_R_IN2, CH_R_PWM = 12, 13, 14

    def __init__(
        self,
        update_rate: float | None = None,
        *,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        pca: Any | None = None,
    ):
        del update_rate  # Kept for compatibility with the retired ROS wrappers.
        self.config_path = Path(config_path)
        self._lock = threading.RLock()
        self._closed = False
        self._owns_pca = pca is None
        self.i2c = None

        config = self._load_config()
        i2c_config = config.get("i2c", {})
        if not isinstance(i2c_config, dict):
            i2c_config = {}
        self.bus_number = int(i2c_config.get("bus", 1))
        self.address = int(i2c_config.get("pca9685_address", 0x70))
        self.frequency = int(i2c_config.get("pwm_frequency", 50))
        self._motor_inverted = motor_inversion_flags(config.get("motors"))
        self._motor_max_speed = self._load_motor_max_speeds(config.get("motors"))
        self._servos = self._load_servos(config.get("servos"))

        if pca is None:
            try:
                from adafruit_pca9685 import PCA9685
            except ImportError as exc:
                raise RuntimeError(
                    "Ubuntu I2C 模式需要安装 adafruit-circuitpython-pca9685"
                ) from exc
            self.i2c = _SMBusI2C(self.bus_number)
            try:
                self.pca = PCA9685(self.i2c, address=self.address)
            except Exception:
                self.i2c.close()
                raise
        else:
            self.pca = pca

        self.pca.frequency = self.frequency
        self._initialize_outputs()

    def _load_config(self) -> dict[str, Any]:
        try:
            with self.config_path.open("r", encoding="utf-8") as stream:
                config = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"读取硬件配置失败: {self.config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError("硬件配置根节点必须是对象")
        return config

    @staticmethod
    def _load_motor_max_speeds(motors: object) -> dict[str, float]:
        limits = {"left": 100.0, "right": 100.0}
        if not isinstance(motors, list):
            return limits
        for motor in motors:
            if not isinstance(motor, dict):
                continue
            side = {"track_l": "left", "track_r": "right"}.get(motor.get("name"))
            if side:
                limits[side] = max(0.0, min(100.0, float(motor.get("max_speed", 100))))
        return limits

    @staticmethod
    def _load_servos(servos: object) -> dict[str, dict[str, int]]:
        loaded: dict[str, dict[str, int]] = {}
        if not isinstance(servos, list):
            return loaded
        for servo in servos:
            if not isinstance(servo, dict):
                continue
            try:
                channel = int(servo["id"])
                name = str(servo["name"])
                limit_1 = int(servo["limit_1"])
                limit_2 = int(servo["limit_2"])
                initial = int(servo["init"])
            except (KeyError, TypeError, ValueError):
                continue
            if not name or not 0 <= channel <= 8:
                continue
            low, high = sorted((limit_1, limit_2))
            loaded[name] = {
                "channel": channel,
                "low": max(0, low),
                "high": min(65535, high),
                "init": max(low, min(high, initial)),
            }
        return loaded

    def _initialize_outputs(self) -> None:
        # Stop the tracks before enabling any servo output.
        for channel in range(9, 15):
            self.pca.channels[channel].duty_cycle = self.MOTOR_LOW
        for servo in self._servos.values():
            self.pca.channels[servo["channel"]].duty_cycle = servo["init"]

    @classmethod
    def _angle_to_duty(cls, angle: float) -> int:
        angle = max(0.0, min(180.0, float(angle)))
        return int(
            cls.SERVO_MIN_DUTY
            + angle / 180.0 * (cls.SERVO_MAX_DUTY - cls.SERVO_MIN_DUTY)
        )

    @staticmethod
    def _throttle_to_duty(throttle: float) -> int:
        throttle = max(0.0, min(100.0, float(throttle)))
        return int(throttle / 100.0 * 65535)

    def set_pwm(self, name: str, pwm: float) -> bool:
        servo = self._servos.get(name)
        if servo is None or self._closed:
            return False
        value = max(servo["low"], min(servo["high"], int(pwm)))
        with self._lock:
            self.pca.channels[servo["channel"]].duty_cycle = value
        return True

    def set_angle(self, name: str, angle: float) -> bool:
        return self.set_pwm(name, self._angle_to_duty(angle))

    def set_motor(self, side: str, action: int, throttle: float) -> bool:
        side = {"track_l": "left", "track_r": "right"}.get(side, side)
        channels = {
            "left": (self.CH_L_IN1, self.CH_L_IN2, self.CH_L_PWM),
            "right": (self.CH_R_IN1, self.CH_R_IN2, self.CH_R_PWM),
        }.get(side)
        if channels is None or self._closed:
            return False

        try:
            action = int(action)
            throttle = float(throttle)
        except (TypeError, ValueError):
            action, throttle = 0, 0.0
        action = apply_direction_inversion(action, self._motor_inverted[side])
        throttle = max(0.0, min(self._motor_max_speed[side], throttle))
        in1_channel, in2_channel, pwm_channel = channels

        if action == 1:
            in1, in2 = self.MOTOR_HIGH, self.MOTOR_LOW
        elif action == 2:
            in1, in2 = self.MOTOR_LOW, self.MOTOR_HIGH
        else:
            in1, in2, throttle = self.MOTOR_LOW, self.MOTOR_LOW, 0.0

        with self._lock:
            # Remove drive power before changing direction inputs.
            self.pca.channels[pwm_channel].duty_cycle = self.MOTOR_LOW
            self.pca.channels[in1_channel].duty_cycle = in1
            self.pca.channels[in2_channel].duty_cycle = in2
            self.pca.channels[pwm_channel].duty_cycle = self._throttle_to_duty(throttle)
        return True

    def stop(self) -> None:
        if self._closed:
            return
        with self._lock:
            self.set_motor("left", 0, 0)
            self.set_motor("right", 0, 0)
            self._closed = True
            try:
                self.pca.deinit()
            finally:
                if self.i2c is not None:
                    self.i2c.close()
