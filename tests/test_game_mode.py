"""Tests for the dependency-free game lifecycle state machine."""

import unittest

from services.game_mode import GameMode, GameModeController, InvalidGameTransition


class GameModeControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = GameModeController()

    def test_game_mode_requires_safe_transition_before_input_is_enabled(self):
        self.assertEqual(self.controller.mode, GameMode.ROBOT)
        self.assertTrue(self.controller.policy.robot_input)
        self.assertTrue(self.controller.policy.recording)

        self.controller.request_enter()
        self.assertTrue(self.controller.policy.motors_must_stop)
        self.assertFalse(self.controller.policy.robot_input)
        self.assertFalse(self.controller.policy.game_input)
        self.assertFalse(self.controller.policy.recording)

        self.controller.game_surface_ready()
        self.assertEqual(self.controller.mode, GameMode.MENU)
        self.assertTrue(self.controller.policy.game_input)
        self.assertFalse(self.controller.policy.game_audio)

        self.controller.start_game()
        self.assertEqual(self.controller.mode, GameMode.PLAYING)
        self.assertTrue(self.controller.policy.game_audio)
        self.assertTrue(self.controller.policy.screenshot_analysis)

    def test_disconnect_pauses_without_restoring_robot_input(self):
        self.controller.request_enter()
        self.controller.game_surface_ready()
        self.controller.start_game()

        self.controller.pause_for_fault()

        self.assertEqual(self.controller.mode, GameMode.PAUSED)
        self.assertFalse(self.controller.policy.robot_input)
        self.assertFalse(self.controller.policy.recording)
        self.assertTrue(self.controller.policy.game_input)
        self.assertTrue(self.controller.policy.motors_must_stop)

    def test_finished_game_returns_to_menu_policy(self):
        self.controller.request_enter()
        self.controller.game_surface_ready()
        self.controller.start_game()

        self.controller.return_to_menu()

        self.assertEqual(self.controller.mode, GameMode.MENU)
        self.assertTrue(self.controller.policy.game_input)
        self.assertFalse(self.controller.policy.game_audio)
        self.assertFalse(self.controller.policy.screenshot_analysis)

    def test_exit_only_restores_robot_after_teardown_acknowledgement(self):
        self.controller.request_enter()
        self.controller.game_surface_ready()
        self.controller.start_game()
        self.controller.request_exit()

        self.assertFalse(self.controller.policy.robot_input)
        self.assertFalse(self.controller.policy.recording)
        self.controller.robot_surface_ready()
        self.assertEqual(self.controller.mode, GameMode.ROBOT)
        self.assertTrue(self.controller.policy.robot_input)
        self.assertTrue(self.controller.policy.recording)

    def test_invalid_transitions_are_rejected(self):
        with self.assertRaises(InvalidGameTransition):
            self.controller.start_game()
        with self.assertRaises(InvalidGameTransition):
            self.controller.robot_surface_ready()
        with self.assertRaises(InvalidGameTransition):
            self.controller.return_to_menu()

    def test_controller_disconnect_can_abort_entering(self):
        self.controller.request_enter()
        self.controller.request_exit()
        self.controller.robot_surface_ready()

        self.assertEqual(self.controller.mode, GameMode.ROBOT)


if __name__ == "__main__":
    unittest.main()
