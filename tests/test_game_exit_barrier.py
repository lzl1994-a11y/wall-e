import unittest

from services.game_exit_barrier import GameExitBarrier


class GameExitBarrierTests(unittest.TestCase):
    def test_initial_session_finished_completes_immediately(self):
        barrier = GameExitBarrier()
        self.assertTrue(barrier.mark_session_finished())

    def test_prepared_session_finished_waits_for_audio(self):
        barrier = GameExitBarrier()
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_session_finished())

    def test_prepared_audio_finished_first_then_session_finished(self):
        barrier = GameExitBarrier()
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_audio_finished())
        self.assertTrue(barrier.mark_session_finished())

    def test_prepared_session_finished_first_then_audio_finished(self):
        barrier = GameExitBarrier()
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_session_finished())
        self.assertTrue(barrier.mark_audio_finished())

    def test_audio_finished_without_prepare_is_noop(self):
        barrier = GameExitBarrier()
        self.assertFalse(barrier.mark_audio_finished())
        self.assertTrue(barrier.mark_session_finished())

    def test_reset_allows_clean_reuse(self):
        barrier = GameExitBarrier()
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_session_finished())
        self.assertTrue(barrier.mark_audio_finished())
        barrier.reset()

        # Round 2: without trailing audio
        self.assertTrue(barrier.mark_session_finished())
        barrier.reset()

        # Round 3: audio finishes first then session
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_audio_finished())
        self.assertTrue(barrier.mark_session_finished())

    def test_idempotent_repeated_calls(self):
        barrier = GameExitBarrier()
        # Repeated session finished on initial state
        self.assertTrue(barrier.mark_session_finished())
        self.assertTrue(barrier.mark_session_finished())

        barrier.reset()
        barrier.prepare_audio_end()
        # Repeated session finished while waiting for audio
        self.assertFalse(barrier.mark_session_finished())
        self.assertFalse(barrier.mark_session_finished())

        # Repeated audio finished after session finished
        self.assertTrue(barrier.mark_audio_finished())
        self.assertTrue(barrier.mark_audio_finished())
        self.assertTrue(barrier.mark_session_finished())

        barrier.reset()
        # Repeated audio finished without prepare
        self.assertFalse(barrier.mark_audio_finished())
        self.assertFalse(barrier.mark_audio_finished())

        # Prepare, then repeated audio finished before session finished
        barrier.prepare_audio_end()
        self.assertFalse(barrier.mark_audio_finished())
        self.assertFalse(barrier.mark_audio_finished())
        # Then session finished
        self.assertTrue(barrier.mark_session_finished())
        self.assertTrue(barrier.mark_session_finished())


if __name__ == "__main__":
    unittest.main()
