import unittest

from services.dialog.dialog_turn import DialogTurnController


class DialogTurnControllerTests(unittest.TestCase):
    def _controller(self):
        return DialogTurnController(turn_id_factory=lambda: "turn-1")

    def test_stream_skips_correction_prefix_and_emits_after_two_punctuations(self):
        controller = self._controller()

        waiting = controller.consume_chunk("you: 你好")
        spoken = controller.consume_chunk("\n你好！很高兴见到你。")

        self.assertEqual(waiting.turn_id, "turn-1")
        self.assertIsNone(waiting.tts_text)
        self.assertEqual(spoken.tts_text, "你好！很高兴见到你。")

    def test_reply_flushes_tail_and_preserves_dialog_fields(self):
        controller = self._controller()
        controller.consume_chunk("you: hi\nai: 你好")

        reply = controller.finish_reply("you: 你好\nai: 很高兴见到你")

        self.assertEqual(reply.turn_id, "turn-1")
        self.assertEqual(reply.corrected_text, "你好")
        self.assertEqual(reply.ai_text, "很高兴见到你")
        self.assertEqual(reply.tts_tail, "ai: 你好")

    def test_finish_resets_turn_and_next_turn_gets_a_new_id(self):
        ids = iter(("turn-1", "turn-2"))
        controller = DialogTurnController(turn_id_factory=lambda: next(ids))

        self.assertEqual(controller.ensure_turn_id(), "turn-1")
        self.assertEqual(controller.finish(), "turn-1")
        self.assertEqual(controller.ensure_turn_id(), "turn-2")

    def test_game_turn_id_can_be_supplied_by_the_adapter(self):
        controller = self._controller()
        controller.set_turn_id("game-1234")

        self.assertEqual(controller.finish(), "game-1234")
