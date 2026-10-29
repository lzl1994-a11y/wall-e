"""Startup prompts must gate recognition without taking over audio policies."""

import importlib
import queue
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

from services.audio.audio_pipeline import AudioPipeline
# Load framework dependencies before patch.dict restores the ROS test stubs.
# FastMCP installs an import hook that must retain its real supporting modules.
from services.llm.llm_service import LLMService  # noqa: F401
from services.llm.voice_chat_service import VoiceChatService  # noqa: F401
from services.speech.stt_service import STTService


class StartupVoiceReadinessTests(unittest.TestCase):
    def _load_node(self, name):
        class String:
            def __init__(self, data=''):
                self.data = data

        class Node:
            def __init__(self, *_args, **_kwargs):
                self.create_publisher = MagicMock()
                self.create_subscription = MagicMock()
                self.create_timer = MagicMock()
                self.get_logger = MagicMock(return_value=MagicMock())

        rclpy_node = types.ModuleType('rclpy.node')
        rclpy_node.Node = Node
        messages = types.ModuleType('std_msgs.msg')
        messages.String = messages.UInt8MultiArray = String
        modules = {'rclpy': types.ModuleType('rclpy'), 'rclpy.node': rclpy_node,
                   'std_msgs': types.ModuleType('std_msgs'), 'std_msgs.msg': messages}
        sys.modules.pop(name, None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module(name)
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def _stt(self):
        module = self._load_node('nodes.stt_ros_node')
        node = module.STTNode.__new__(module.STTNode)
        node._game_active = node._llm_busy = node._recording_paused = False
        node._remote_connected = False
        node._startup_prompt_gate = MagicMock()
        node._startup_prompt_gate.active.return_value = True
        node.stt_engine = MagicMock()
        node.publisher_ = MagicMock()
        node.get_logger = lambda: MagicMock()
        return node

    def _pipe(self, *, raw=False, shared=False):
        pipe = AudioPipeline.__new__(AudioPipeline)
        pipe._is_running, pipe._is_paused = True, False
        pipe._raw_only, pipe._shared_capture = raw, shared
        pipe._startup_prompt_gate = MagicMock()
        pipe._startup_prompt_gate.active.return_value = True
        pipe._startup_microphone_ready = False
        pipe._external_session_active = pipe._external_turn_active = False
        pipe.audio_queue = queue.Queue()
        pipe.on_raw_pcm = MagicMock()
        return pipe

    def test_stt_prompt_gate_is_independent_of_dialogue_busy(self):
        node = self._stt()
        node._sync_recording()
        node.stt_engine.pause.assert_called_once()
        self.assertFalse(node._llm_busy)
        node.on_speech_detected('瓦力准备好了，可以叫我了')
        node.publisher_.publish.assert_not_called()
        node._startup_prompt_gate.active.return_value = False
        node._sync_recording()
        node.stt_engine.resume.assert_called_once()

    def test_valid_asr_submission_is_announced_before_voice_text(self):
        node = self._stt()
        node._startup_prompt_gate.active.return_value = False
        events = []
        node._startup_recognition_pub = MagicMock()
        node.publisher_.get_subscription_count.return_value = 1
        node._startup_recognition_pub.publish.side_effect = lambda msg: events.append(('state', msg.data))
        node.publisher_.publish.side_effect = lambda msg: events.append(('voice', msg.data))
        node._on_recognition_state(True)
        node.on_speech_detected('低头')
        node._on_recognition_state(False)
        self.assertEqual(events, [('state', 'busy'), ('state', 'submitted'),
                                  ('voice', '低头'), ('state', 'submitted')])
        node._on_llm_busy(types.SimpleNamespace(data='busy'))
        self.assertFalse(node._startup_utterance_submitted)
        self.assertTrue(node._llm_busy)
        node._on_recognition_state(False)
        self.assertEqual(events[-1], ('state', 'idle'))

    def test_asr_result_before_llm_subscription_does_not_hold_ready_forever(self):
        node = self._stt()
        node._startup_prompt_gate.active.return_value = False
        node.publisher_.get_subscription_count.return_value = 0
        node._startup_recognition_pub = MagicMock()
        events = []
        node._startup_recognition_pub.publish.side_effect = lambda msg: events.append(('state', msg.data))
        node.publisher_.publish.side_effect = lambda msg: events.append(('voice', msg.data))
        node._on_recognition_state(True)
        node.on_speech_detected('低头')
        node._on_recognition_state(False)
        self.assertEqual(events, [('state', 'busy'), ('voice', '低头'), ('state', 'idle')])
        self.assertFalse(node._startup_utterance_submitted)

    def test_failed_or_suppressed_voice_submission_releases_recognition(self):
        for suppressed in (False, True):
            with self.subTest(suppressed=suppressed):
                node = self._stt()
                node._startup_prompt_gate.active.return_value = False
                node._startup_recognition_pub = MagicMock()
                node.publisher_.get_subscription_count.return_value = 1
                node._on_recognition_state(True)
                if suppressed:
                    node._game_active = True
                    node.on_speech_detected('低头')
                else:
                    node.publisher_.publish.side_effect = RuntimeError('publisher unavailable')
                    with self.assertRaises(RuntimeError):
                        node.on_speech_detected('低头')
                node._on_recognition_state(False)
                self.assertEqual(node._startup_recognition_pub.publish.call_args.args[0].data, 'idle')
                self.assertFalse(getattr(node, '_startup_utterance_submitted', False))

    def test_finishing_prompt_never_overrides_other_recording_blockers(self):
        for blocker in ('_game_active', '_llm_busy', '_remote_connected'):
            with self.subTest(blocker=blocker):
                node = self._stt()
                node._sync_recording()
                setattr(node, blocker, True)
                node._startup_prompt_gate.active.return_value = False
                node._sync_recording()
                node.stt_engine.resume.assert_not_called()
                self.assertTrue(getattr(node, blocker))
                setattr(node, blocker, False)
                node._sync_recording()
                node.stt_engine.resume.assert_called_once()

    def test_prompt_pcm_does_not_enter_local_recognition_queue(self):
        pipe = self._pipe()
        with patch('services.audio.audio_pipeline.mark_startup_ready') as mark:
            pipe._queue_processed_pcm(b'\x00\x00')
            pipe._queue_processed_pcm(b'\x01\x00')
        mark.assert_called_once_with('microphone')
        self.assertEqual(pipe.on_raw_pcm.call_count, 2)
        self.assertTrue(pipe.audio_queue.empty())
        pipe._startup_prompt_gate.active.return_value = False
        pipe._queue_processed_pcm(b'\x02\x00')
        self.assertEqual(pipe.audio_queue.get_nowait(), b'\x02\x00')

    def test_raw_capture_and_remote_input_remain_available_during_prompt(self):
        pipe = self._pipe(raw=True)
        with patch('services.audio.audio_pipeline.mark_startup_ready'):
            pipe._queue_processed_pcm(b'\x01\x00')
        pipe.on_raw_pcm.assert_called_once_with(b'\x01\x00')
        pipe._startup_prompt_gate.active.assert_not_called()
        for external in ('_external_session_active', '_external_turn_active'):
            with self.subTest(external=external):
                pipe = self._pipe(shared=True)
                setattr(pipe, external, True)
                pipe._queue_processed_pcm(b'\x01\x00')
                self.assertEqual(pipe.audio_queue.get_nowait(), b'\x01\x00')

    def test_shared_consumer_cannot_claim_physical_microphone_readiness(self):
        pipe = self._pipe(shared=True)
        with patch('services.audio.audio_pipeline.mark_startup_ready') as mark:
            pipe._queue_processed_pcm(b'\x00\x00')
        mark.assert_not_called()

    def test_prompt_never_reaches_wake_detector_and_model_is_reused(self):
        pipe = self._pipe()
        pipe._paused_event = threading.Event()
        pipe._wake_lock = threading.Lock()
        pipe._ww = MagicMock()
        pipe._ww.enabled = True
        pipe._ww.check.return_value = True
        detector = pipe._ww
        self.assertFalse(pipe._check_wake_word(b'prompt'))
        detector.check.assert_not_called()
        pipe._startup_prompt_gate.active.return_value = False
        self.assertTrue(pipe._check_wake_word(b'user'))
        detector.check.assert_called_once_with(b'user')
        self.assertIs(pipe._ww, detector)

    def test_capture_disconnect_invalidates_microphone_readiness(self):
        pipe = self._pipe()
        pipe._startup_microphone_ready = True
        pipe._audio_stream_lock = threading.Lock()
        stream = pipe._audio_stream = MagicMock()
        with patch('services.audio.audio_pipeline.clear_startup_ready') as clear:
            pipe._close_audio_stream()
        clear.assert_called_once_with('microphone')
        self.assertFalse(pipe._startup_microphone_ready)
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_stt_ready_requires_successful_engine_start(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                module = self._load_node('nodes.stt_ros_node')
                engine = MagicMock()
                engine.start.side_effect = RuntimeError('unavailable') if fail else None
                with (patch.object(module, 'STTService', return_value=engine),
                      patch.object(module, 'StartupPromptGate') as gate,
                      patch.object(module, 'mark_startup_ready') as ready,
                      patch.dict('os.environ', {'WALI_SHARED_MICROPHONE': '0', 'WALI_STARTUP_STATUS_DIR': ''})):
                    gate.return_value.active.return_value = False
                    module.STTNode()
                self.assertEqual(ready.call_count, 0 if fail else 1)

    def test_llm_ready_follows_successful_client_and_worker_start(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                module = self._load_node('nodes.llm_ros_node')
                events = []

                def service():
                    events.append('client')
                    if fail:
                        raise RuntimeError('unavailable')
                    return MagicMock()

                thread = MagicMock()
                thread.start.side_effect = lambda: events.append('worker')
                with (patch.object(module, 'LLMService', side_effect=service),
                      patch.object(module, 'create_wali_task_action_client', side_effect=RuntimeError('test')),
                      patch.object(module, 'load_tft_preview_settings'),
                      patch.object(module, 'TftPreviewClient'),
                      patch.object(module, 'RollingVoiceDebugStore'),
                      patch.object(module.threading, 'Thread', return_value=thread),
                      patch.object(module, 'mark_startup_ready', side_effect=lambda: events.append('ready')),
                      patch.dict('os.environ', {'WALI_SHARED_MICROPHONE': '0', 'WALI_STARTUP_STATUS_DIR': ''})):
                    module.LLMBrainNode()
                self.assertEqual(events, ['client'] if fail else ['client', 'worker', 'ready'])


class StartupRecognitionStateTests(unittest.TestCase):
    def _service(self, *, streaming=False):
        service = STTService.__new__(STTService)
        service._awake = True
        service._awake_lock = threading.Lock()
        service._awake_timer = None
        service._awake_timer_generation = 0
        service._streaming_active = streaming
        service._streaming_lock = threading.Lock()
        service.asr_adapter = MagicMock()
        service.on_sentence_received = MagicMock()
        service.on_speech_end = MagicMock()
        service._reset_awake_timer = MagicMock()
        service._voice_debug = None
        service.on_recognition_state = MagicMock()
        return service

    def test_recognition_stays_busy_across_finish_stream_and_wav_fallback(self):
        service = self._service(streaming=True)
        finish_started, finish_release = threading.Event(), threading.Event()
        fallback_started, fallback_release = threading.Event(), threading.Event()

        def finish():
            finish_started.set()
            self.assertTrue(finish_release.wait(3))
            raise RuntimeError('stream failed')

        def fallback(*_args):
            fallback_started.set()
            self.assertTrue(fallback_release.wait(3))
            return '识别成功'

        service.asr_adapter.finish_stream.side_effect = finish
        service.asr_adapter.recognize.side_effect = fallback
        worker = threading.Thread(target=service._on_sentence, args=(b'\0\0' * 1600,), daemon=True)
        worker.start()
        try:
            self.assertTrue(finish_started.wait(2))
            self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True])
            finish_release.set()
            self.assertTrue(fallback_started.wait(2))
            self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True])
        finally:
            finish_release.set()
            fallback_release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True, False])
        service.on_sentence_received.assert_called_once_with('识别成功')

    def test_early_return_and_failures_always_release_recognition(self):
        for mode in ('not_awake', 'recognize_error', 'vad_callback_error'):
            with self.subTest(mode=mode):
                service = self._service()
                if mode == 'not_awake':
                    service._awake = False
                elif mode == 'recognize_error':
                    service.asr_adapter.recognize.side_effect = RuntimeError('ASR failed')
                else:
                    service.on_speech_end.side_effect = RuntimeError('callback failed')
                if mode == 'vad_callback_error':
                    with self.assertRaises(RuntimeError):
                        service._on_sentence(b'\0\0')
                else:
                    service._on_sentence(b'\0\0')
                self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True, False])
                self.assertEqual(service._recognition_inflight, 0)

    def test_concurrent_recognitions_do_not_emit_idle_until_last_returns(self):
        service = self._service()
        counter_lock = threading.Lock()
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        calls = []

        def recognize(*_args):
            with counter_lock:
                index = len(calls)
                calls.append(index)
            entered[index].set()
            self.assertTrue(release[index].wait(3))
            return '识别成功'

        service.asr_adapter.recognize.side_effect = recognize
        workers = [threading.Thread(target=service._on_sentence, args=(b'\0\0',), daemon=True) for _ in range(2)]
        for worker in workers:
            worker.start()
        try:
            self.assertTrue(all(event.wait(2) for event in entered))
            self.assertEqual(service._recognition_inflight, 2)
            release[0].set()
            # The matching worker may be either list element, so await the
            # counter's one remaining call without assuming thread order.
            for _ in range(100):
                if service._recognition_inflight == 1:
                    break
                threading.Event().wait(0.005)
            self.assertEqual(service._recognition_inflight, 1)
            self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True])
        finally:
            for event in release:
                event.set()
            for worker in workers:
                worker.join(3)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual([call.args[0] for call in service.on_recognition_state.call_args_list], [True, False])

    def test_feedback_callback_failure_cannot_fail_recognition(self):
        service = self._service()
        service.on_recognition_state.side_effect = RuntimeError('optional publisher unavailable')
        service.asr_adapter.recognize.return_value = '识别成功'
        service._on_sentence(b'\0\0')
        service.on_sentence_received.assert_called_once_with('识别成功')
        self.assertEqual(service._recognition_inflight, 0)
