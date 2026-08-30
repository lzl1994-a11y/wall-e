import unittest

from services.fc_input import (
    ABS_HAT0X,
    BTN_EAST,
    BTN_NORTH,
    BTN_SELECT,
    BTN_SOUTH,
    BTN_START,
    BTN_WEST,
    EV_ABS,
    EV_KEY,
    FcInputMapper,
    FcJoystickMapper,
    FcKeyChange,
)


class FcInputMapperTests(unittest.TestCase):
    def setUp(self):
        self.mapper = FcInputMapper()

    def test_face_buttons_use_fceux_default_a_and_b_keys(self):
        cases = {
            BTN_SOUTH: "F",
            BTN_NORTH: "F",
            BTN_EAST: "D",
            BTN_WEST: "D",
        }
        for code, key in cases.items():
            with self.subTest(code=code):
                self.assertEqual(
                    self.mapper.translate(EV_KEY, code, 1),
                    [FcKeyChange(key, True)],
                )
                self.assertEqual(
                    self.mapper.translate(EV_KEY, code, 0),
                    [FcKeyChange(key, False)],
                )

    def test_select_and_start(self):
        self.assertEqual(
            self.mapper.translate(EV_KEY, BTN_SELECT, 1),
            [FcKeyChange("S", True)],
        )
        self.assertEqual(
            self.mapper.translate(EV_KEY, BTN_START, 1),
            [FcKeyChange("Return", True)],
        )

    def test_hat_releases_old_direction_before_pressing_new_one(self):
        self.assertEqual(
            self.mapper.translate(EV_ABS, ABS_HAT0X, -1),
            [FcKeyChange("KP_4", True)],
        )
        self.assertEqual(
            self.mapper.translate(EV_ABS, ABS_HAT0X, 1),
            [FcKeyChange("KP_4", False), FcKeyChange("KP_6", True)],
        )
        self.assertEqual(
            self.mapper.translate(EV_ABS, ABS_HAT0X, 0),
            [FcKeyChange("KP_6", False)],
        )

    def test_repeats_and_unmapped_robot_controls_are_ignored(self):
        self.assertEqual(self.mapper.translate(EV_KEY, BTN_SOUTH, 2), [])
        self.assertEqual(self.mapper.translate(EV_KEY, 310, 1), [])
        self.assertEqual(self.mapper.translate(EV_ABS, 0, 255), [])


class FcJoystickMapperTests(unittest.TestCase):
    def test_maps_confirmed_buttons_and_dpad_axes(self):
        mapper = FcJoystickMapper()
        self.assertEqual(mapper.translate(1, 0, 1), [FcKeyChange("F", True)])
        self.assertEqual(mapper.translate(1, 1, 1), [FcKeyChange("D", True)])
        self.assertEqual(mapper.translate(2, 6, -32767), [FcKeyChange("KP_4", True)])
        self.assertEqual(mapper.translate(2, 6, 0), [FcKeyChange("KP_4", False)])

    def test_ignores_analog_axes_and_understands_initial_events(self):
        mapper = FcJoystickMapper()
        self.assertEqual(mapper.translate(2, 0, 32767), [])
        self.assertEqual(mapper.translate(0x81, 9, 1), [FcKeyChange("Return", True)])


if __name__ == "__main__":
    unittest.main()
