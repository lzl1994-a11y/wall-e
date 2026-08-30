import unittest

from services.libretro_fc import (
    LibretroJoypad,
    RETRO_DEVICE_ID_JOYPAD_A,
    RETRO_DEVICE_ID_JOYPAD_B,
    RETRO_DEVICE_ID_JOYPAD_LEFT,
    RETRO_DEVICE_ID_JOYPAD_START,
    RETRO_DEVICE_JOYPAD,
)


class LibretroJoypadTests(unittest.TestCase):
    def test_maps_adapter_keys_to_the_first_fc_controller(self):
        pad = LibretroJoypad()
        pad.set_key("F", True)
        pad.set_key("D", True)
        pad.set_key("KP_4", True)
        pad.set_key("Return", True)
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A), 1)
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B), 1)
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_LEFT), 1)
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_START), 1)

    def test_release_and_other_devices_are_ignored(self):
        pad = LibretroJoypad()
        pad.set_key("F", True)
        pad.set_key("F", False)
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A), 0)
        self.assertEqual(pad.state(1, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A), 0)

    def test_close_releases_all_buttons(self):
        pad = LibretroJoypad()
        pad.set_key("F", True)
        pad.close()
        self.assertEqual(pad.state(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A), 0)


if __name__ == "__main__":
    unittest.main()
