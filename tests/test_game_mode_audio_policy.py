"""Game-mode audio policy stays state-only and does not alter AudioPipeline."""

import unittest

from services.game_mode import GameMode, GameModeController


class GameModeAudioPolicyTests(unittest.TestCase):
    def test_entering_game_mode_disables_recording_before_surface_is_ready(self):
        controller = GameModeController()

        controller.request_enter()

        self.assertEqual(controller.mode, GameMode.ENTERING)
        self.assertFalse(controller.policy.recording)
        self.assertTrue(controller.policy.motors_must_stop)

    def test_robot_mode_restores_recording_only_after_teardown_ack(self):
        controller = GameModeController()
        controller.request_enter()
        controller.game_surface_ready()
        controller.request_exit()

        self.assertFalse(controller.policy.recording)

        controller.robot_surface_ready()

        self.assertEqual(controller.mode, GameMode.ROBOT)
        self.assertTrue(controller.policy.recording)


if __name__ == "__main__":
    unittest.main()
