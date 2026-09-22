import unittest

from services.dialog.dialog_output import DialogOutputController


class DialogOutputControllerTests(unittest.TestCase):
    def test_wake_completion_resumes_only_when_no_tts_is_pending(self):
        controller = DialogOutputController()
        self.assertTrue(controller.start_wake("wake-1").accepted)
        controller.mark_tts_queued()

        wake_done = controller.finish_wake("wake-1")
        tts_done = controller.finish_tts_playback()

        self.assertTrue(wake_done.accepted)
        self.assertFalse(wake_done.schedule_capture_resume)
        self.assertTrue(tts_done.schedule_capture_resume)

    def test_tts_completion_waits_for_matching_wake_completion(self):
        controller = DialogOutputController()
        controller.start_wake("wake-1")
        controller.mark_tts_queued()

        tts_done = controller.finish_tts_playback()
        wake_done = controller.finish_wake("wake-1")

        self.assertFalse(tts_done.schedule_capture_resume)
        self.assertTrue(wake_done.schedule_capture_resume)

    def test_stale_wake_completion_is_ignored(self):
        controller = DialogOutputController()
        controller.start_wake("wake-current")

        stale = controller.finish_wake("wake-previous")

        self.assertFalse(stale.accepted)
        self.assertTrue(controller.wake_response_active)
        self.assertEqual(controller.wake_request_id, "wake-current")

    def test_shutdown_blocks_new_wakes_and_capture_resume(self):
        controller = DialogOutputController()
        controller.shutdown()

        self.assertFalse(controller.start_wake("wake-1").accepted)
        self.assertFalse(controller.can_resume_capture())
