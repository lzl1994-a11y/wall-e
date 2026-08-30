import unittest

from services.game_protocol import (
    decode_game_frame,
    encode_game_frame,
    encode_game_request,
    game_is_active,
)


class GameProtocolTests(unittest.TestCase):
    def test_raw_frame_round_trip_preserves_layout(self):
        raw = bytes(range(32))
        packet = encode_game_frame(raw, 2, 4, 8)

        self.assertEqual(decode_game_frame(packet), (raw, 2, 4, 8))
        self.assertIsNone(decode_game_frame(packet[:-1]))

    def test_game_state_and_request_are_compact_json(self):
        self.assertTrue(game_is_active('{"mode":"menu"}'))
        self.assertFalse(game_is_active('{"mode":"robot"}'))
        self.assertEqual(
            encode_game_request("toggle", controller="/dev/input/event2"),
            '{"request":"toggle","controller":"/dev/input/event2"}',
        )


if __name__ == "__main__":
    unittest.main()
