import unittest

from services.game_hotkey import ButtonChordHold, StartSelectHold


class StartSelectHoldTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.hotkey = StartSelectHold(hold_seconds=2.0, clock=lambda: self.now)

    def test_emits_once_after_both_buttons_are_held(self):
        self.hotkey.set_start(True)
        self.now += 0.3
        self.hotkey.set_select(True)
        self.now += 1.99
        self.assertFalse(self.hotkey.poll())
        self.now += 0.01
        self.assertTrue(self.hotkey.poll())

    def test_generic_chord_supports_x_and_y_semantics(self):
        now = [3.0]
        hotkey = ButtonChordHold(hold_seconds=2.0, clock=lambda: now[0])
        hotkey.set_first(True)
        hotkey.set_second(True)
        now[0] = 5.0

        self.assertTrue(hotkey.poll())
        self.assertFalse(self.hotkey.poll())

    def test_release_before_deadline_cancels_the_hotkey(self):
        self.hotkey.set_start(True)
        self.hotkey.set_select(True)
        self.now += 1.0
        self.hotkey.set_select(False)
        self.now += 2.0
        self.assertFalse(self.hotkey.poll())

    def test_release_arms_a_new_long_press(self):
        self.hotkey.set_start(True)
        self.hotkey.set_select(True)
        self.now += 2.0
        self.assertTrue(self.hotkey.poll())
        self.hotkey.set_start(False)
        self.hotkey.set_start(True)
        self.now += 2.0
        self.assertTrue(self.hotkey.poll())


if __name__ == "__main__":
    unittest.main()
