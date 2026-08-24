import unittest
from unittest.mock import patch

from services.action_intent_guard import validate_action_call


class ActionIntentGuardTests(unittest.TestCase):
    def assertAllowed(self, text, name, arguments):
        allowed, reason = validate_action_call(text, name, arguments)
        self.assertTrue(allowed, reason)

    def assertRejected(self, text, name, arguments, expected_reason=None):
        allowed, reason = validate_action_call(text, name, arguments)
        self.assertFalse(allowed)
        if expected_reason:
            self.assertEqual(reason, expected_reason)

    def test_explicit_and_polite_action_commands_are_allowed(self):
        self.assertAllowed(
            "向左转头。",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
        )
        self.assertAllowed(
            "你能帮我向左转一下头吗？",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
        )
        self.assertAllowed(
            "能不能向前走一下？",
            "move_chassis",
            {"direction": "forward", "duration": 1},
        )

    def test_capability_question_story_and_past_event_are_rejected(self):
        self.assertRejected(
            "你能转头吗？",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
            "non_command_context",
        )
        self.assertRejected(
            "给我讲一个机器人挥手的故事。",
            "play_sequence",
            {"sequence_name": "wave_hello"},
            "non_command_context",
        )
        self.assertRejected(
            "我刚才向左看了。",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
            "non_command_context",
        )
        self.assertRejected(
            "瓦力有没有转头的能力？",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
            "non_command_context",
        )
        self.assertRejected(
            "你现在会走路吗？",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )

    def test_tracking_stop_only_allows_idle(self):
        self.assertAllowed(
            "别再跟着我了。",
            "set_tracking_mode",
            {"mode": "idle"},
        )
        self.assertRejected(
            "别再跟着我了。",
            "set_tracking_mode",
            {"mode": "follow_me"},
            "stop_command_mismatch",
        )
        self.assertAllowed(
            "我不想让你跟着我。",
            "set_tracking_mode",
            {"mode": "idle"},
        )
        self.assertAllowed(
            "跟着我就不用了。",
            "set_tracking_mode",
            {"mode": "idle"},
        )
        self.assertRejected(
            "你能停止跟随吗？",
            "set_tracking_mode",
            {"mode": "idle"},
            "non_command_context",
        )
        self.assertRejected(
            "停止跟随是什么意思？",
            "set_tracking_mode",
            {"mode": "idle"},
            "non_command_context",
        )
        self.assertRejected(
            "为什么关闭视觉跟踪？",
            "set_vision_gate",
            {"enabled": False},
            "non_command_context",
        )

    def test_negated_physical_action_is_never_executed(self):
        self.assertRejected(
            "不要向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "negated_action",
        )
        self.assertRejected(
            "别转头。",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
            "negated_action",
        )
        self.assertRejected(
            "不要左转。",
            "move_chassis",
            {"direction": "left", "duration": 1},
            "negated_action",
        )
        self.assertRejected(
            "停止向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "negated_action",
        )
        self.assertRejected(
            "别拍照。",
            "inspect_camera",
            {"question": ""},
            "negated_action",
        )
        self.assertRejected(
            "不需要拍照。",
            "inspect_camera",
            {"question": ""},
            "negated_action",
        )
        self.assertRejected(
            "不要打开视觉跟踪。",
            "set_vision_gate",
            {"enabled": True},
            "negated_action",
        )
        self.assertRejected(
            "不要停止跟随。",
            "set_tracking_mode",
            {"mode": "idle"},
            "negated_action",
        )
        self.assertRejected(
            "不要关闭视觉跟踪。",
            "set_vision_gate",
            {"enabled": False},
            "negated_action",
        )
        self.assertRejected(
            "别拍张照。",
            "inspect_camera",
            {"question": ""},
            "negated_action",
        )
        self.assertRejected(
            "不想向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "negated_action",
        )
        self.assertRejected(
            "不想打开视觉跟踪。",
            "set_vision_gate",
            {"enabled": True},
            "negated_action",
        )
        self.assertRejected(
            "停止跟随不用了。",
            "set_tracking_mode",
            {"mode": "idle"},
            "negated_action",
        )
        self.assertRejected(
            "关闭视觉跟踪算了。",
            "set_vision_gate",
            {"enabled": False},
            "negated_action",
        )
        self.assertAllowed(
            "停止跟随，为什么还在跟我？",
            "set_tracking_mode",
            {"mode": "idle"},
        )
        self.assertAllowed(
            "关闭视觉跟踪，为什么还开着？",
            "set_vision_gate",
            {"enabled": False},
        )

    def test_plain_chat_third_party_and_explicit_conflicts_fail_closed(self):
        self.assertRejected(
            "你好。",
            "move_chassis",
            {"direction": "forward", "duration": 3},
            "non_command_context",
        )
        self.assertRejected(
            "后退。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "argument_conflict",
        )
        self.assertRejected(
            "跟着我。",
            "set_tracking_mode",
            {"mode": "idle"},
            "argument_conflict",
        )
        self.assertRejected(
            "他正在向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )
        self.assertRejected(
            "让小明向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )
        self.assertRejected(
            "小王正在向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )
        self.assertRejected(
            "昨天瓦力向前走了。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )
        self.assertRejected(
            "“向前走”是个命令句。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "non_command_context",
        )
        self.assertRejected(
            "向前走就不用了。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "negated_action",
        )

    def test_direction_duration_and_mode_must_match_the_user_command(self):
        self.assertAllowed(
            "后退一秒。",
            "move_chassis",
            {"direction": "backward", "duration": 1},
        )
        self.assertRejected(
            "向前走两秒。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "argument_conflict",
        )
        self.assertRejected(
            "向前走。",
            "move_chassis",
            {"direction": "forward", "duration": 3},
            "argument_conflict",
        )
        self.assertRejected(
            "往前挪一点。",
            "move_chassis",
            {"direction": "backward", "duration": 1},
            "argument_conflict",
        )
        self.assertAllowed(
            "看着我。",
            "set_tracking_mode",
            {"mode": "look_at_me"},
        )
        self.assertRejected(
            "看着我。",
            "set_tracking_mode",
            {"mode": "follow_me"},
            "argument_conflict",
        )

    def test_model_is_positive_router_for_flexible_commands(self):
        self.assertAllowed(
            "看看你前面有什么。",
            "inspect_camera",
            {"question": "看看你前面有什么。"},
        )
        self.assertAllowed(
            "你从看我手里拿的是什么？",
            "inspect_camera",
            {"question": "你从看我手里拿的是什么？"},
        )
        self.assertAllowed(
            "往前挪一点。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
        )
        self.assertAllowed(
            "脑袋往左转转。",
            "play_sequence",
            {"sequence_name": "turn_head_left"},
        )
        self.assertAllowed(
            "盯住我别乱看。",
            "set_tracking_mode",
            {"mode": "look_at_me"},
        )
        self.assertRejected(
            "别往前挪。",
            "move_chassis",
            {"direction": "forward", "duration": 1},
            "negated_action",
        )
        self.assertAllowed(
            "做个开心的表情。",
            "express_emotion",
            {"emotion": "happy"},
        )
        self.assertRejected(
            "我今天很开心。",
            "express_emotion",
            {"emotion": "happy"},
            "non_command_context",
        )

    def test_malformed_or_out_of_range_arguments_fail_closed(self):
        self.assertRejected("向前走", "move_chassis", {}, "invalid_arguments")
        self.assertRejected(
            "向前走十秒",
            "move_chassis",
            {"direction": "forward", "duration": 10},
            "invalid_arguments",
        )
        self.assertRejected("做个动作", "unknown_tool", {}, "unknown_tool")
        with patch("services.action_intent_guard._SEQUENCE_NAMES", frozenset()):
            self.assertRejected(
                "向左转头",
                "play_sequence",
                {"sequence_name": "turn_head_left"},
                "invalid_arguments",
            )


if __name__ == "__main__":
    unittest.main()
