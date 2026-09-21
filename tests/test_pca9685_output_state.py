import unittest

from services.pca9685_output_state import Pca9685OutputState


class Pca9685OutputStateTests(unittest.TestCase):
    def test_default_state_and_initial_dirty(self):
        # 默认全部通道初始化为 90° 对应值: int(1638 + (8192 - 1638) * 90 / 180) = 4915
        # 电机通道 9-14 强制置零
        state = Pca9685OutputState()
        self.assertTrue(state.dirty)

        expected = [4915] * 9 + [0] * 6
        self.assertEqual(state.encode(), "pca9685:" + ",".join(str(v) for v in expected))

    def test_config_init_overrides_and_motor_channels_forced_zero(self):
        # 接收配置提供的 servo init 值覆盖对应通道，即使提供了 9-14 也最终被强制归零
        servo_inits = {
            0: 2800,
            4: 5000,
            8: 8000,
            9: 9999,
            12: 12345,
        }
        state = Pca9685OutputState(servo_inits)
        self.assertTrue(state.dirty)

        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[0], 2800)
        self.assertEqual(values[4], 5000)
        self.assertEqual(values[8], 8000)
        # 9-14 强制置零
        self.assertEqual(values[9:15], [0, 0, 0, 0, 0, 0])

    def test_pwm_and_angle_writing_and_formula(self):
        state = Pca9685OutputState()
        state.mark_published()
        self.assertFalse(state.dirty)

        # set_pwm 写入与 dirty
        state.set_pwm(0, 3000)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[0], 3000)

        # set_angle: 严格公式 int(1638 + (8192 - 1638) * angle / 180)，不要 clamp
        state.mark_published()
        # 0 度
        state.set_angle(1, 0)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[1], 1638)

        # 90 度
        state.mark_published()
        state.set_angle(1, 90)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[1], int(1638 + (8192 - 1638) * 90 / 180))

        # 180 度
        state.mark_published()
        state.set_angle(1, 180)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[1], 8192)

        # 负角度与超过 180 度（不要 clamp）
        state.mark_published()
        state.set_angle(2, -10)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[2], int(1638 + (8192 - 1638) * -10 / 180))

        state.mark_published()
        state.set_angle(3, 200)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[3], int(1638 + (8192 - 1638) * 200 / 180))

    def test_same_value_does_not_set_dirty(self):
        state = Pca9685OutputState({0: 3000})
        state.mark_published()
        self.assertFalse(state.dirty)

        # 写入相同的 PWM 值，不触发 dirty
        state.set_pwm(0, 3000)
        self.assertFalse(state.dirty)

        # 写入相同的 angle 产生相同的 PWM 值，不触发 dirty
        state.set_angle(4, 90)  # 4 号通道初始即为 90°
        self.assertFalse(state.dirty)

    def test_motor_actions_and_throttle_conversion(self):
        state = Pca9685OutputState()
        state.mark_published()

        # 左电机正转: action=1 -> HIGH (65535), LOW (0); throttle=50 -> int(50 / 100.0 * 65535) = 32767
        # left 对应 9, 10, 11
        state.set_motor("left", 1, 50)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[9], 65535)
        self.assertEqual(values[10], 0)
        self.assertEqual(values[11], 32767)

        # 右电机反转: action=2 -> LOW (0), HIGH (65535); throttle=25 -> int(25 / 100.0 * 65535) = 16383
        # right 对应 12, 13, 14
        state.mark_published()
        state.set_motor("right", 2, 25)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[12], 0)
        self.assertEqual(values[13], 65535)
        self.assertEqual(values[14], 16383)

        # 写入相同电机指令不重新 dirty
        state.mark_published()
        state.set_motor("right", 2, 25)
        self.assertFalse(state.dirty)

        # 停止: action=0 或其他 -> LOW, LOW 且 throttle 强制为 0
        state.mark_published()
        state.set_motor("left", 0, 50)
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[9:12], [0, 0, 0])

        state.mark_published()
        state.set_motor("right", 99, 100)  # 其他 action
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[12:15], [0, 0, 0])

    def test_stop_motors_encode_and_lifecycle(self):
        state = Pca9685OutputState()
        state.set_motor("left", 1, 80)
        state.set_motor("right", 2, 80)
        state.mark_published()
        self.assertFalse(state.dirty)

        # stop_motors 停止两侧电机
        state.stop_motors()
        self.assertTrue(state.dirty)
        values = [int(v) for v in state.encode().split(":", 1)[1].split(",")]
        self.assertEqual(values[9:15], [0, 0, 0, 0, 0, 0])

        # 编码格式验证
        encoded = state.encode()
        self.assertTrue(encoded.startswith("pca9685:"))
        payload_parts = encoded[len("pca9685:"):].split(",")
        self.assertEqual(len(payload_parts), 15)

        # 再次 stop_motors，电机原本就全停，不产生 dirty
        state.mark_published()
        state.stop_motors()
        self.assertFalse(state.dirty)

        # 后续真实变化再次触发 dirty
        state.set_pwm(0, 1234)
        self.assertTrue(state.dirty)


if __name__ == "__main__":
    unittest.main()
