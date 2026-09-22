"""Unit tests for pure joystick button policy service."""

import unittest

from services.motion.joystick_button_policy import (
    BTN_A,
    BTN_B,
    BTN_L1,
    BTN_R1,
    BTN_X,
    BTN_Y,
    HAT_X,
    HAT_Y,
    JoystickButtonPolicy,
)


class JoystickButtonPolicyTests(unittest.TestCase):
    def setUp(self):
        self.monotonic_time = 100.0
        self.wall_time = 1000.0
        self.policy = JoystickButtonPolicy(
            hold_seconds=2.0,
            chord_clock=lambda: self.monotonic_time,
        )

    def test_x_single_press_and_release_sends_wave_hello(self):
        # Press X
        d1 = self.policy.handle_key(BTN_X, 1, now=self.wall_time)
        self.assertIsNone(d1.action)
        self.assertTrue(self.policy.x_down)
        self.assertFalse(self.policy.y_down)

        # Release X
        d2 = self.policy.handle_key(BTN_X, 0, now=self.wall_time)
        self.assertEqual(d2.action, "wave_hello")
        self.assertEqual(d2.timer_updates, {})
        self.assertFalse(self.policy.x_down)
        self.assertFalse(self.policy.chord_fired)

    def test_y_single_press_and_release_sends_raise_hand(self):
        # Press Y
        d1 = self.policy.handle_key(BTN_Y, 1, now=self.wall_time)
        self.assertIsNone(d1.action)
        self.assertTrue(self.policy.y_down)
        self.assertFalse(self.policy.x_down)

        # Release Y
        d2 = self.policy.handle_key(BTN_Y, 0, now=self.wall_time)
        self.assertEqual(d2.action, "raise_hand")
        self.assertEqual(d2.timer_updates, {})
        self.assertFalse(self.policy.y_down)
        self.assertFalse(self.policy.chord_fired)

    def test_brief_chord_release_order_maintains_existing_behavior(self):
        # Case 1: Press X, then press Y, then release X, then release Y
        self.policy.handle_key(BTN_X, 1)
        self.policy.handle_key(BTN_Y, 1)
        self.assertTrue(self.policy.x_down)
        self.assertTrue(self.policy.y_down)

        # Release X while Y is still held: other_down (Y) is True -> no action
        d1 = self.policy.handle_key(BTN_X, 0)
        self.assertIsNone(d1.action)

        # Release Y: other_down (X) is now False, chord did not fire -> triggers raise_hand
        d2 = self.policy.handle_key(BTN_Y, 0)
        self.assertEqual(d2.action, "raise_hand")
        self.assertFalse(self.policy.x_down)
        self.assertFalse(self.policy.y_down)

        # Case 2: Press Y, then press X, then release Y, then release X
        self.policy.handle_key(BTN_Y, 1)
        self.policy.handle_key(BTN_X, 1)

        d3 = self.policy.handle_key(BTN_Y, 0)
        self.assertIsNone(d3.action)

        d4 = self.policy.handle_key(BTN_X, 0)
        self.assertEqual(d4.action, "wave_hello")
        self.assertFalse(self.policy.x_down)
        self.assertFalse(self.policy.y_down)

    def test_long_press_chord_fires_toggle_once(self):
        # Press both X and Y
        self.policy.handle_key(BTN_X, 1)
        self.policy.handle_key(BTN_Y, 1)

        # Advance 1.9s -> should not fire
        self.monotonic_time += 1.9
        self.assertFalse(self.policy.poll_game_toggle())
        self.assertFalse(self.policy.chord_fired)

        # Advance 0.1s (total 2.0s) -> fires once
        self.monotonic_time += 0.1
        self.assertTrue(self.policy.poll_game_toggle())
        self.assertTrue(self.policy.chord_fired)

        # Poll again -> does not fire repeatedly
        self.assertFalse(self.policy.poll_game_toggle())

    def test_release_after_chord_fired_suppresses_normal_actions(self):
        # Press both and trigger chord
        self.policy.handle_key(BTN_X, 1)
        self.policy.handle_key(BTN_Y, 1)
        self.monotonic_time += 2.0
        self.assertTrue(self.policy.poll_game_toggle())
        self.assertTrue(self.policy.chord_fired)

        # Release X -> suppressed
        d1 = self.policy.handle_key(BTN_X, 0)
        self.assertIsNone(d1.action)
        self.assertTrue(self.policy.chord_fired)

        # Release Y -> suppressed, and now both are up so chord_fired clears
        d2 = self.policy.handle_key(BTN_Y, 0)
        self.assertIsNone(d2.action)
        self.assertFalse(self.policy.chord_fired)

        # New press and release of X after clearing -> normal action works again
        self.policy.handle_key(BTN_X, 1)
        d3 = self.policy.handle_key(BTN_X, 0)
        self.assertEqual(d3.action, "wave_hello")

    def test_game_mode_suppression_and_state_tracking(self):
        # X/Y down states are still tracked in game mode
        self.policy.handle_key(BTN_X, 1, game_active=True)
        self.assertTrue(self.policy.x_down)
        self.policy.handle_key(BTN_Y, 1, game_active=True)
        self.assertTrue(self.policy.y_down)

        # Chord toggle does not fire when game mode is active
        self.monotonic_time += 3.0
        self.assertFalse(self.policy.poll_game_toggle(game_active=True))

        # Releasing X/Y in game mode does not send wave_hello / raise_hand
        d_x = self.policy.handle_key(BTN_X, 0, game_active=True)
        self.assertIsNone(d_x.action)
        d_y = self.policy.handle_key(BTN_Y, 0, game_active=True)
        self.assertIsNone(d_y.action)

        # Other buttons (A, B, L1, R1) are completely suppressed
        self.assertIsNone(self.policy.handle_key(BTN_A, 1, game_active=True).action)
        self.assertIsNone(self.policy.handle_key(BTN_B, 1, game_active=True).action)
        self.assertEqual(self.policy.handle_key(BTN_L1, 1, game_active=True).timer_updates, {})
        self.assertEqual(self.policy.handle_key(BTN_R1, 1, game_active=True).timer_updates, {})

    def test_key_repeat_value_2_and_release_value_0_for_other_keys(self):
        # value=2 on X/Y is ignored and does not alter down state
        d_x2 = self.policy.handle_key(BTN_X, 2)
        self.assertIsNone(d_x2.action)
        self.assertFalse(self.policy.x_down)

        d_y2 = self.policy.handle_key(BTN_Y, 2)
        self.assertIsNone(d_y2.action)
        self.assertFalse(self.policy.y_down)

        # value=2 on A, B, L1, R1 does not trigger
        self.assertIsNone(self.policy.handle_key(BTN_A, 2).action)
        self.assertIsNone(self.policy.handle_key(BTN_B, 2).action)
        self.assertEqual(self.policy.handle_key(BTN_L1, 2).timer_updates, {})
        self.assertEqual(self.policy.handle_key(BTN_R1, 2).timer_updates, {})

        # value=0 on A, B, L1, R1 does not trigger
        self.assertIsNone(self.policy.handle_key(BTN_A, 0).action)
        self.assertIsNone(self.policy.handle_key(BTN_B, 0).action)
        self.assertEqual(self.policy.handle_key(BTN_L1, 0).timer_updates, {})
        self.assertEqual(self.policy.handle_key(BTN_R1, 0).timer_updates, {})

    def test_a_and_b_actions(self):
        dA = self.policy.handle_key(BTN_A, 1)
        self.assertEqual(dA.action, "happy_dance")
        self.assertEqual(dA.timer_updates, {})

        dB = self.policy.handle_key(BTN_B, 1)
        self.assertEqual(dB.action, "sad_react")
        self.assertEqual(dB.timer_updates, {})

    def test_l1_and_r1_eyebrow_timers(self):
        dL1 = self.policy.handle_key(BTN_L1, 1, now=1000.0, auto_reset_delay=3.0)
        self.assertIsNone(dL1.action)
        self.assertEqual(dL1.timer_updates, {"eyebrow_l": 1003.0})

        dR1 = self.policy.handle_key(BTN_R1, 1, now=1000.0, auto_reset_delay=5.0)
        self.assertIsNone(dR1.action)
        self.assertEqual(dR1.timer_updates, {"eyebrow_r": 1005.0})

    def test_hat_x_and_y_directions_and_clearing(self):
        now = 500.0
        delay = 3.0

        # HAT_X left (-1)
        dx_left = self.policy.handle_hat(HAT_X, -1, now=now, auto_reset_delay=delay)
        self.assertEqual(dx_left.timer_updates, {"arm_l": 503.0})

        # HAT_X right (1)
        dx_right = self.policy.handle_hat(HAT_X, 1, now=now, auto_reset_delay=delay)
        self.assertEqual(dx_right.timer_updates, {"arm_r": 503.0})

        # HAT_Y up (-1): updates both arms
        dy_up = self.policy.handle_hat(HAT_Y, -1, now=now, auto_reset_delay=delay)
        self.assertEqual(dy_up.timer_updates, {"arm_l": 503.0, "arm_r": 503.0})

        # HAT_Y down (1): clears both arms to 0.0
        dy_down = self.policy.handle_hat(HAT_Y, 1, now=now, auto_reset_delay=delay)
        self.assertEqual(dy_down.timer_updates, {"arm_l": 0.0, "arm_r": 0.0})

        # Neutral / invalid hat values produce no timer updates
        self.assertEqual(self.policy.handle_hat(HAT_X, 0).timer_updates, {})
        self.assertEqual(self.policy.handle_hat(HAT_Y, 0).timer_updates, {})
        self.assertEqual(self.policy.handle_hat(HAT_X, 2).timer_updates, {})
        self.assertEqual(self.policy.handle_hat(HAT_Y, -2).timer_updates, {})

    def test_hat_unaffected_by_game_mode(self):
        now = 200.0
        d = self.policy.handle_hat(HAT_X, -1, now=now, auto_reset_delay=3.0)
        self.assertEqual(d.timer_updates, {"arm_l": 203.0})

    def test_unknown_codes_have_no_effect(self):
        key_decision = self.policy.handle_key(999, 1, now=100.0)
        hat_decision = self.policy.handle_hat(999, -1, now=100.0)
        self.assertIsNone(key_decision.action)
        self.assertEqual(key_decision.timer_updates, {})
        self.assertIsNone(hat_decision.action)
        self.assertEqual(hat_decision.timer_updates, {})


if __name__ == "__main__":
    unittest.main()
