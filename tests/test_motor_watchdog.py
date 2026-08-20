import unittest

from services.motor_watchdog import MotorWatchdog


class MotorWatchdogTests(unittest.TestCase):
    def test_trips_once_after_timeout_and_recovers_on_refresh(self):
        now = [20.0]
        watchdog = MotorWatchdog(timeout_sec=0.3, clock=lambda: now[0])

        now[0] += 0.29
        self.assertFalse(watchdog.poll())
        now[0] += 0.01
        self.assertTrue(watchdog.poll())
        self.assertFalse(watchdog.poll())

        self.assertTrue(watchdog.refresh())
        now[0] += 0.3
        self.assertTrue(watchdog.poll())


if __name__ == "__main__":
    unittest.main()
