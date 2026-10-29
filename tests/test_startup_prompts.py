import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from services.audio.startup_prompts import (
    POST_PROMPT_GUARD_SECONDS,
    PROMPT_TIMEOUT_SECONDS,
    StartupPromptController,
    StartupPromptGate,
    clear_startup_ready,
    mark_startup_ready,
)


class StartupPromptTests(unittest.TestCase):
    def test_bundled_clips_use_mixer_pcm_format_and_leading_guard_silence(self):
        root = Path(__file__).resolve().parents[1] / 'assets'
        for name in ('startup_begin.wav', 'startup_ready.wav'):
            with self.subTest(name=name), wave.open(str(root / name), 'rb') as source:
                self.assertEqual((source.getframerate(), source.getnchannels(), source.getsampwidth()), (48000, 1, 2))
                self.assertGreater(source.getnframes(), 48000)
                self.assertEqual(source.readframes(4800), b'\0\0' * 4800)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.session = self.root / 'session-one'
        self.assets = self.root / 'assets'
        self.session.mkdir()
        self.assets.mkdir()
        for name in ('startup_begin.wav', 'startup_ready.wav'):
            with wave.open(str(self.assets / name), 'wb') as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(48000)
                target.writeframes(b'\x01\x00' * 4800)
        self.required = ['audio_playback', 'llm', 'stt', 'microphone']
        self.environment = patch.dict(os.environ, {
            'WALI_STARTUP_STATUS_DIR': str(self.session),
            'WALI_STARTUP_REQUIRED': json.dumps(self.required),
            'WALI_STARTUP_COMPONENT': 'llm',
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.clock = patch('services.audio.startup_prompts.time.monotonic', return_value=100.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def controller(self):
        return StartupPromptController(assets_root=self.assets)

    def test_skipping_optional_feedback_preserves_tail_then_stops_polling(self):
        controller = self.controller()
        request, _, _ = controller.next_prompt()
        controller.skip_ready()
        self.assertTrue(controller.finished)
        self.assertIsNone(controller.pending_request)
        self.assertTrue(StartupPromptGate().active())
        self.advance()
        self.assertFalse(StartupPromptGate().active())
        self.assertIsNone(controller.next_prompt())
        self.assertFalse(controller.finish(request))

    def advance(self, seconds=POST_PROMPT_GUARD_SECONDS + 0.01):
        self.now.return_value += seconds

    def ready(self, names=None):
        for name in names or self.required:
            self.assertTrue(mark_startup_ready(name))

    def finish_begin(self, controller):
        request_id, cue, _pcm = controller.next_prompt()
        self.assertEqual(cue, 'startup_begin')
        self.assertTrue(controller.finish(request_id))
        self.advance()

    def test_disabled_standalone_nodes_do_not_touch_files(self):
        with patch.dict(os.environ, {}, clear=True):
            controller = StartupPromptController(assets_root=self.root / 'missing')
            self.assertFalse(controller.enabled)
            self.assertIsNone(controller.next_prompt())
            self.assertFalse(controller.finish('none'))
            self.assertFalse(StartupPromptGate().active())
            self.assertFalse(mark_startup_ready())
            clear_startup_ready('llm')
        self.assertEqual(list(self.session.iterdir()), [])

    def test_begin_is_first_and_ready_requires_every_live_component(self):
        controller = self.controller()
        gate = StartupPromptGate()
        request_id, cue, pcm = controller.next_prompt()
        self.assertEqual((cue, pcm), ('startup_begin', b'\x01\x00' * 4800))
        self.assertEqual(controller.pending_request, request_id)
        self.assertEqual(controller.pending_cue, cue)
        self.assertTrue(gate.active())
        self.ready(self.required[:-1])
        self.assertIsNone(controller.next_prompt())
        self.assertFalse(controller.finish('unrelated-network-prompt'))
        self.assertTrue(gate.active())
        controller.finish(request_id)
        self.assertIsNone(controller.pending_request)
        self.assertTrue(gate.active(), 'the acoustic tail must keep recognition paused')
        self.advance()
        self.assertFalse(gate.active())
        self.assertIsNone(controller.next_prompt(), 'missing microphone cannot be called ready')
        mark_startup_ready('microphone')
        ready_id, cue, _pcm = controller.next_prompt()
        self.assertEqual(cue, 'startup_ready')
        self.assertTrue(gate.active())
        controller.finish(ready_id)
        self.assertTrue(gate.active())
        self.advance()
        self.assertFalse(gate.active())
        with patch('services.audio.startup_prompts._read_json', side_effect=AssertionError('continued filesystem polling')):
            self.assertFalse(gate.active())
        self.assertIsNone(controller.next_prompt())

    def test_audio_process_restart_does_not_replay_clips(self):
        first = self.controller()
        self.finish_begin(first)
        restarted = self.controller()
        self.assertIsNone(restarted.next_prompt())
        self.ready()
        request_id, cue, _pcm = restarted.next_prompt()
        self.assertEqual(cue, 'startup_ready')
        self.assertIsNone(self.controller().next_prompt())
        restarted.finish(request_id)
        self.assertIsNone(self.controller().next_prompt())

    def test_old_session_readiness_cannot_complete_a_new_session(self):
        self.ready()
        new_session = self.root / 'session-two'
        new_session.mkdir()
        with patch.dict(os.environ, {'WALI_STARTUP_STATUS_DIR': str(new_session)}):
            controller = self.controller()
            self.finish_begin(controller)
            self.assertIsNone(controller.next_prompt())

    def test_dead_component_blocks_ready_and_clear_invalidates_live_readiness(self):
        controller = self.controller()
        self.finish_begin(controller)
        self.ready()
        (self.session / 'llm.ready').write_text('{"pid":999999999}', encoding='utf-8')
        self.assertIsNone(controller.next_prompt())
        self.assertTrue(mark_startup_ready())
        clear_startup_ready('microphone')
        self.assertIsNone(controller.next_prompt())
        mark_startup_ready('microphone')
        clear_startup_ready()
        self.assertIsNone(controller.next_prompt())
        mark_startup_ready()
        self.assertEqual(controller.next_prompt()[1], 'startup_ready')

    def test_lost_completion_times_out_recognition_gate_without_announcing_ready(self):
        controller = self.controller()
        request_id, _cue, _pcm = controller.next_prompt()
        self.ready()
        gate = StartupPromptGate()
        self.assertTrue(gate.active())
        self.advance(PROMPT_TIMEOUT_SECONDS + 1)
        self.assertFalse(gate.active())
        self.assertIsNone(controller.next_prompt(), 'a timeout is not a playback completion')
        self.assertTrue(controller.cancel(request_id))
        self.advance()
        self.assertEqual(controller.next_prompt()[1], 'startup_ready')

    def test_dead_player_releases_gate(self):
        controller = self.controller()
        controller.next_prompt()
        path = self.session / 'prompt-active.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        state['pid'] = 999999999
        path.write_text(json.dumps(state), encoding='utf-8')
        self.assertFalse(StartupPromptGate().active())

    def test_cancel_ready_releases_gate_and_does_not_replay(self):
        controller = self.controller()
        self.finish_begin(controller)
        self.ready()
        request_id, _cue, _pcm = controller.next_prompt()
        self.assertTrue(controller.cancel(request_id))
        self.assertIsNone(controller.pending_request)
        self.assertIsNone(controller.pending_cue)
        self.assertTrue(StartupPromptGate().active())
        self.advance()
        self.assertFalse(StartupPromptGate().active())
        self.assertIsNone(self.controller().next_prompt())

    def test_duplicate_or_old_completion_does_not_extend_or_release_a_new_gate(self):
        controller = self.controller()
        old_id, _cue, _pcm = controller.next_prompt()
        controller.finish(old_id)
        self.advance()
        self.ready()
        new_id, _cue, _pcm = controller.next_prompt()
        before = (self.session / 'prompt-active.json').read_text(encoding='utf-8')
        self.assertFalse(controller.finish(old_id))
        self.assertEqual((self.session / 'prompt-active.json').read_text(encoding='utf-8'), before)
        self.assertEqual(controller.pending_request, new_id)
        self.assertTrue(StartupPromptGate().active())

    def test_stopping_session_does_not_issue_begin_or_ready(self):
        stop = self.session / 'stopping'
        stop.touch()
        controller = self.controller()
        self.assertIsNone(controller.next_prompt())
        stop.unlink()
        self.finish_begin(controller)
        self.ready()
        stop.touch()
        self.assertIsNone(controller.next_prompt())

    def test_invalid_component_names_cannot_write_outside_session(self):
        for name in ('../outside', '/outside', '', 'llm.ready', 'a\\outside'):
            self.assertFalse(mark_startup_ready(name))
            clear_startup_ready(name)
        self.assertEqual(list(self.session.iterdir()), [])

    def test_invalid_required_metadata_disables_controller(self):
        for required in ('not-json', '[]', '{}', '["../outside"]'):
            with patch.dict(os.environ, {'WALI_STARTUP_REQUIRED': required}):
                self.assertFalse(self.controller().enabled)

    def test_missing_or_wrong_format_assets_fail_before_issuing_any_clip(self):
        path = self.assets / 'startup_ready.wav'
        with wave.open(str(path), 'wb') as target:
            target.setnchannels(2)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(b'\x01\x00' * 32)
        with self.assertRaises(ValueError):
            self.controller()
        self.assertEqual(list(self.session.iterdir()), [])

    def test_malformed_gate_never_leaves_recognition_muted(self):
        path = self.session / 'prompt-active.json'
        for deadline in (float('inf'), float('nan'), True, None, 9999999):
            path.write_text(json.dumps({'pid': os.getpid(), 'phase': 'playing', 'deadline': deadline}), encoding='utf-8')
            self.assertFalse(StartupPromptGate().active())
        path.write_text('not-json', encoding='utf-8')
        self.assertFalse(StartupPromptGate().active())
