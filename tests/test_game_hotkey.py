import unittest

from services.game_hotkey import StartSelectHold


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
