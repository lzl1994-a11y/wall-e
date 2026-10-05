import threading
import unittest
from unittest.mock import MagicMock, patch

from services.llm.voice_chat_service import VoiceChatService, _State


class VoiceChatCaptureGuardTests(unittest.TestCase):
    def _service(self, state):
        service = VoiceChatService.__new__(VoiceChatService)
        service._state = state
        service._state_lock = threading.Lock()
        service._pipe = MagicMock()
        service._last_llm_activity = 0.0
        service._cancel_llm = threading.Event()
        service._remote_session_active = False
        service._remote_session_end_requested = False
        service.on_llm_done = None
        service.on_llm_timeout = None
        return service

    def test_output_playback_mutes_capture_until_completion(self):
        service = self._service(_State.AWAKE)

        service.begin_output_playback()

        self.assertEqual(service._state, _State.SPEAKING)
        service._pipe.pause.assert_called_once_with()
        service._pipe.resume.assert_not_called()

        self.assertTrue(service.complete_output_playback())
        self.assertEqual(service._state, _State.AWAKE)
        service._pipe.resume.assert_called_once_with()

    def test_llm_done_waits_in_speaking_state(self):
        service = self._service(_State.LLM_PENDING)
        service.on_llm_done = MagicMock()

        service._llm_done()

        self.assertEqual(service._state, _State.SPEAKING)
        service._pipe.resume.assert_not_called()
        service.on_llm_done.assert_called_once_with()

    def test_valid_sentence_claims_turn_and_mutes_capture_before_dispatch(self):
        service = self._service(_State.AWAKE)
        service._dispatch_llm = MagicMock()
        pcm = b"\x00\x00" * (VoiceChatService.SAMPLE_RATE // 5)

        with patch("os.makedirs"), patch("shutil.copy2"):
            service._on_sentence(pcm)

        self.assertEqual(service._state, _State.LLM_PENDING)
        service._pipe.pause.assert_called_once_with()
        service._dispatch_llm.assert_called_once()

    def test_short_noise_does_not_mute_capture(self):
        service = self._service(_State.AWAKE)
        service._dispatch_llm = MagicMock()
        pcm = b"\x00\x00" * (VoiceChatService.SAMPLE_RATE // 10)

        service._on_sentence(pcm)

        self.assertEqual(service._state, _State.AWAKE)
        service._pipe.pause.assert_not_called()
        service._dispatch_llm.assert_not_called()

    def test_stale_playback_completion_does_not_resume_idle_capture(self):
        service = self._service(_State.IDLE)

        self.assertFalse(service.complete_output_playback())
        service._pipe.resume.assert_not_called()

    def test_remote_session_end_waits_for_audio_pipeline_tail(self):
        service = self._service(_State.AWAKE)
        service._remote_session_active = True
        service._remote_session_end_requested = False
        service._pipe.end_external_session.return_value = True

        self.assertTrue(service.end_remote_session())
        self.assertTrue(service._remote_session_active)
        self.assertTrue(service._remote_session_end_requested)
        self.assertEqual(service._state, _State.AWAKE)

        service._on_external_session_end()

        self.assertFalse(service._remote_session_active)
        self.assertFalse(service._remote_session_end_requested)
        self.assertEqual(service._state, _State.IDLE)

    def test_remote_llm_timeout_keeps_the_call_audio_gate_alive(self):
        service = self._service(_State.LLM_PENDING)
        service._remote_session_active = True

        service._on_timeout()

        self.assertEqual(service._state, _State.AWAKE)
        service._pipe.set_awake.assert_not_called()
        service._pipe.resume.assert_not_called()

    def test_stale_llm_completion_does_not_close_a_reopened_turn(self):
        service = self._service(_State.AWAKE)
        service.on_llm_done = MagicMock()

        service._llm_done()

        service.on_llm_done.assert_not_called()
        self.assertEqual(service._state, _State.AWAKE)

    def test_remote_speech_cancels_an_inflight_llm_turn(self):
        service = self._service(_State.LLM_PENDING)
        service._remote_session_active = True
        service.on_remote_barge_in = MagicMock()

        service._on_speech_start(b"initial")

        self.assertEqual(service._state, _State.AWAKE)
        self.assertTrue(service._cancel_llm.is_set())
        service.on_remote_barge_in.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
