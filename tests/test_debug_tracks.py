import unittest

from tools.diagnostics.debug_tracks import encode_packet, initial_state, motion_state


class DebugTracksTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "servos": [{"id": 0, "init": 3000}, {"id": 8, "init": 8000}],
            "motors": [
                {"name": "track_l", "invert_direction": True},
                {"name": "track_r", "invert_direction": False},
            ],
        }

    def test_initial_state_keeps_servo_initial_values_and_stops_tracks(self):
        state = initial_state(self.config)
        self.assertEqual(state[0], 3000)
        self.assertEqual(state[8], 8000)
        self.assertEqual(state[9:], [0] * 6)

    def test_forward_respects_configured_left_motor_inversion(self):
        state = motion_state(self.config, "forward", 20)
        self.assertEqual(state[9:12], [0, 65535, 13107])
        self.assertEqual(state[12:15], [65535, 0, 13107])
        self.assertEqual(state[0], 3000)
        self.assertEqual(state[8], 8000)

    def test_packet_is_newline_delimited_15_channel_pca_command(self):
        packet = encode_packet([0] * 15)
        self.assertEqual(packet, b"pca9685:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")


if __name__ == "__main__":
    unittest.main()
