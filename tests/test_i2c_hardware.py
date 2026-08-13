import tempfile
import unittest
from pathlib import Path

import yaml

from services.servo_control import ServoControl


class FakeChannel:
    def __init__(self):
        self.duty_cycle = None


class FakePCA:
    def __init__(self):
        self.channels = [FakeChannel() for _ in range(16)]
        self.frequency = None
        self.deinitialized = False

    def deinit(self):
        self.deinitialized = True


class I2CHardwareTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="wali-i2c-")
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            yaml.safe_dump(
                {
                    "i2c": {
                        "bus": 3,
                        "pca9685_address": 0x71,
                        "pwm_frequency": 60,
                    },
                    "servos": [
                        {
                            "id": 0,
                            "name": "eye_r",
                            "limit_1": 2000,
                            "limit_2": 4300,
                            "init": 3000,
                        },
                        {
                            "id": 8,
                            "name": "arm_r",
                            "limit_1": 8000,
                            "limit_2": 4000,
                            "init": 8000,
                        },
                    ],
                    "motors": [
                        {
                            "name": "track_l",
                            "max_speed": 80,
                            "invert_direction": False,
                        },
                        {
                            "name": "track_r",
                            "max_speed": 100,
                            "invert_direction": True,
                        },
                    ],
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.pca = FakePCA()
        self.driver = ServoControl(config_path=self.config_path, pca=self.pca)

    def tearDown(self):
        self.driver.stop()
        self.temp_dir.cleanup()

    def test_initializes_from_configured_channels_and_positions(self):
        self.assertEqual(self.driver.bus_number, 3)
        self.assertEqual(self.driver.address, 0x71)
        self.assertEqual(self.pca.frequency, 60)
        self.assertEqual(self.pca.channels[0].duty_cycle, 3000)
        self.assertEqual(self.pca.channels[8].duty_cycle, 8000)
        self.assertEqual(
            [self.pca.channels[index].duty_cycle for index in range(9, 15)],
            [0, 0, 0, 0, 0, 0],
        )

    def test_raw_pwm_is_clamped_to_servo_limits(self):
        self.assertTrue(self.driver.set_pwm("eye_r", 9999))
        self.assertEqual(self.pca.channels[0].duty_cycle, 4300)
        self.assertTrue(self.driver.set_pwm("eye_r", 1))
        self.assertEqual(self.pca.channels[0].duty_cycle, 2000)
        self.assertFalse(self.driver.set_pwm("missing", 3000))

    def test_motor_direction_inversion_and_speed_limit_are_applied(self):
        self.driver.set_motor("left", 1, 100)
        self.assertEqual(self.pca.channels[9].duty_cycle, 65535)
        self.assertEqual(self.pca.channels[10].duty_cycle, 0)
        self.assertEqual(self.pca.channels[11].duty_cycle, int(0.8 * 65535))

        self.driver.set_motor("right", 1, 50)
        self.assertEqual(self.pca.channels[12].duty_cycle, 0)
        self.assertEqual(self.pca.channels[13].duty_cycle, 65535)
        self.assertEqual(self.pca.channels[14].duty_cycle, int(0.5 * 65535))

    def test_stop_disables_motors_and_releases_pca(self):
        self.driver.set_motor("left", 1, 50)
        self.driver.stop()
        self.assertEqual(
            [self.pca.channels[index].duty_cycle for index in range(9, 15)],
            [0, 0, 0, 0, 0, 0],
        )
        self.assertTrue(self.pca.deinitialized)


if __name__ == "__main__":
    unittest.main()
