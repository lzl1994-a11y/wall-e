from services.game_commentary import GameCommentaryController


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _controller(clock):
    return GameCommentaryController(interval_seconds=lambda: 10.0, clock=clock)


def test_game_mode_pauses_and_resumes_voice_at_robot_boundary():
    clock = _Clock()
    controller = _controller(clock)

    assert controller.set_mode("playing").pause_voice is True
    assert controller.set_mode("inactive").pause_voice is False
    assert controller.set_mode("robot").resume_voice is True
    assert controller.mode == "robot"


def test_due_commentary_reserves_latest_frame_once_and_reschedules():
    clock = _Clock()
    controller = _controller(clock)
    controller.set_mode("playing")
    assert controller.accept_frame("frame") is True

    assert controller.take_due_frame() is None
    clock.now = 10.0
    assert controller.take_due_frame() == "frame"
    assert controller.take_due_frame() is None
    controller.finish_commentary()
    assert controller.take_due_frame() is None
    clock.now = 20.0
    assert controller.take_due_frame() == "frame"


def test_robot_mode_discards_frames_and_prevents_late_commentary_output():
    clock = _Clock()
    controller = _controller(clock)
    controller.set_mode("playing")
    controller.accept_frame("frame")
    clock.now = 10.0
    assert controller.take_due_frame() == "frame"

    decision = controller.set_mode("robot")

    assert decision.clear_frame is True
    assert controller.can_publish_commentary() is False
    controller.finish_commentary()
    assert controller.accept_frame("new-frame") is False
