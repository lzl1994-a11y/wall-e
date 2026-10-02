import unittest

from services.hardware.eyeconfig_rpc import (
    EyeConfigError,
    encode_eye_request,
    normalize_eye_state,
    parse_eye_response,
    validate_eye_command,
)


class EyeConfigProtocolTests(unittest.TestCase):
    def test_validate_documented_commands_and_normalizes_values(self):
        self.assertEqual(
            validate_eye_command("eyeconfig:color=00e5ff"),
            "eyeconfig:color=00E5FF",
        )
        self.assertEqual(validate_eye_command("eyeconfig:brightness=0.850"), "eyeconfig:brightness=0.85")
        self.assertEqual(
            validate_eye_command("eyeaction:look:x=15,y=-8"),
            "eyeaction:look:x=15,y=-8",
        )
        self.assertEqual(validate_eye_command("eyeaction:heart"), "eyeaction:heart")
        self.assertEqual(validate_eye_command("eyeconfig:autoBlink=0"), "eyeconfig:autoBlink=0")

    def test_validate_rejects_out_of_range_or_arbitrary_serial_input(self):
        for command in (
            "eyeconfig:brightness=1.2",
            "eyeconfig:dots=257",
            "eyeaction:look:x=101,y=0",
            "screen_dialog:secret",
            "eyeconfig:color=00E5FF\nopenchat:1",
        ):
            with self.subTest(command=command):
                with self.assertRaises(EyeConfigError):
                    validate_eye_command(command)

    def test_parse_state_accepts_key_value_and_json_payloads(self):
        event = parse_eye_response(
            "EYE:STATE color=00e5ff ringColor=00CFE8 dotColor=00BCD0 "
            "brightness=0.85 autoBlink=1 ring=0 dots=48 mood=flame x=15 y=-8"
        )
        self.assertEqual(event["kind"], "state")
        self.assertEqual(event["fields"]["color"], "00E5FF")
        self.assertEqual(event["fields"]["autoBlink"], True)
        self.assertEqual(event["fields"]["ring"], False)
        self.assertEqual(event["fields"]["lookX"], 15)
        self.assertEqual(event["fields"]["lookY"], -8)
        self.assertEqual(event["fields"]["mood"], "flame")

        json_event = parse_eye_response('EYE:STATE:{"scale":1.25,"glow":22,"mood":"heart"}')
        self.assertEqual(json_event["fields"], {"scale": 1.25, "glow": 22, "mood": "heart"})

    def test_parse_ack_and_error_preserves_device_message(self):
        self.assertEqual(parse_eye_response("EYE:OK")["kind"], "ok")
        error = parse_eye_response("EYE:ERR:unsupported dots")
        self.assertEqual(error["kind"], "err")
        self.assertEqual(error["message"], "unsupported dots")
        self.assertIsNone(parse_eye_response("EYE:STATEFUL nope"))

    def test_malformed_numeric_state_is_ignored_without_breaking_parser(self):
        event = parse_eye_response("EYE:STATE dots=inf mood=heart")
        self.assertEqual(event["fields"], {"mood": "heart"})

    def test_normalize_state_ignores_unknown_keys(self):
        self.assertEqual(
            normalize_eye_state({"centerColor": "#ffffff", "dotCount": "12", "unknown": "x"}),
            {"color": "FFFFFF", "dots": 12},
        )

    def test_encode_request_validates_before_publishing_contract(self):
        body = encode_eye_request("request-1", "eyeconfig:ring=1", "ack")
        self.assertIn('"request_id":"request-1"', body)
        self.assertIn('"command":"eyeconfig:ring=1"', body)
        self.assertIn('"expected":"ack"', body)


if __name__ == "__main__":
    unittest.main()
