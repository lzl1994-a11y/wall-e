from services.dialog_presentation import DialogPresentationController
from services.dialog_turn import ReplyDecision


def test_presentation_builds_stable_wake_timeout_and_reply_payloads():
    controller = DialogPresentationController()

    assert controller.wake_listening()["page"] == "chat"
    assert controller.timed_out()["page"] == "idle"
    reply = controller.reply(
        ReplyDecision("turn-1", "你好", "你好呀", "尾句"),
        [{"status": "completed"}],
    )

    assert reply.tts_tail == "尾句"
    assert reply.screen_payload == {
        "turn_id": "turn-1",
        "corrected_text": "你好",
        "ai_text": "你好呀",
        "actions": [{"status": "completed"}],
        "source": "voice_chat",
    }


def test_presentation_ignores_non_list_action_results():
    result = DialogPresentationController().reply(
        ReplyDecision("turn-1", "", "你好"), {"status": "completed"}
    )

    assert result.screen_payload["actions"] == []
