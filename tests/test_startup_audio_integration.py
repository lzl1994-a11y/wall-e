"""Exercise startup announcements through the existing playback node, no hardware."""

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

# Load third-party dependencies before restoring a temporary ROS module map.
# Some libraries install import hooks whose registrations outlive patch.dict.
import services.audio.mixing_playback_service
import services.audio.esp32_network_prompt
from services.audio.startup_prompts import StartupPromptController, StartupPromptGate, mark_startup_ready
from services.remote.audio_playback_gate import AudioPlaybackGate, REMOTE_AUDIO_EPOCH_LABEL_PREFIX
from services.dialog.dialog_motion_protocol import VAD_SPEECH_STARTED, VAD_SPEECH_ENDED


class StartupAudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='startup-audio-test-')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for name in ('startup_begin.wav', 'startup_ready.wav'):
            with wave.open(str(self.root / name), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(48000)
                output.writeframes(b'\x01\x00' * 100)
        self.required = ['audio_playback', 'llm', 'stt', 'microphone', 'tts_play']
        env = patch.dict(os.environ, {
            'WALI_STARTUP_STATUS_DIR': str(self.root),
            'WALI_STARTUP_REQUIRED': json.dumps(self.required),
            'WALI_STARTUP_COMPONENT': 'audio_playback',
        })
        env.start()
        self.addCleanup(env.stop)
        clock = patch('services.audio.startup_prompts.time.monotonic', return_value=100.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

        class String:
            def __init__(self, data=''):
                self.data = data

        class Node:
            def __init__(self, _name):
                self.params = {}
                self.timers = []

            def declare_parameter(self, name, default):
                self.params[name] = default

            def get_parameter(self, name):
                return types.SimpleNamespace(value=self.params[name])

            def create_publisher(self, *_args, **_kwargs):
                return MagicMock()

            def create_subscription(self, *_args, **_kwargs):
                return MagicMock()

            def create_timer(self, period, callback):
                self.timers.append((period, callback))

            def get_logger(self):
                return MagicMock()

            def destroy_node(self):
                pass

        rclpy_node = types.ModuleType('rclpy.node')
        rclpy_node.Node = Node
        qos = types.ModuleType('rclpy.qos')
        qos.QoSProfile = lambda **kwargs: types.SimpleNamespace(**kwargs)
        qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1)
        qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2)
        msgs = types.ModuleType('std_msgs.msg')
        msgs.String = msgs.UInt8MultiArray = String
        modules = {'rclpy': types.ModuleType('rclpy'), 'rclpy.node': rclpy_node,
                   'rclpy.qos': qos, 'std_msgs': types.ModuleType('std_msgs'),
                   'std_msgs.msg': msgs}
        name = 'nodes.audio_playback_node'
        sys.modules.pop(name, None)
        with patch.dict(sys.modules, modules):
            self.module = importlib.import_module(name)
        self.addCleanup(sys.modules.pop, name, None)

    def node(self):
        node = self.module.AudioPlaybackNode.__new__(self.module.AudioPlaybackNode)
        node._startup_prompts = StartupPromptController(assets_root=self.root)
        node._dialogue_busy = False
        node._startup_input_pending = False
        node._startup_recognition_busy = False
        node._startup_submission_pending = False
        node._remote_audio_gate = AudioPlaybackGate()
        node._player = MagicMock()
        node._player.foreground_active = False
        node._system_done_pub = MagicMock()
        node._dialog_state_pub = MagicMock()
        node._last_tts_sequence = node._received_tts_chunks = node._missing_tts_chunks = 0
        return node

    def test_ready_waits_for_all_live_components_and_only_announces_once(self):
        node = self.node()
        node._poll_startup_prompts()
        beginning = node._startup_prompts.pending_request
        self.assertTrue(beginning.startswith('startup_begin-'))
        self.assertEqual(node._player.play_prompt.call_args.args[1], 'system')
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        node._on_system_complete(beginning)
        self.clock.return_value = 101.0
        for component in self.required[:-1]:
            mark_startup_ready(component)
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        mark_startup_ready(self.required[-1])
        node._poll_startup_prompts()
        ready = node._startup_prompts.pending_request
        self.assertTrue(ready.startswith('startup_ready-'))
        self.assertEqual(node._player.play_prompt.call_count, 2)
        node._on_system_complete(ready)
        self.clock.return_value = 102.0
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 2)
        node._dialog_state_pub.publish.assert_not_called()

    def test_foreground_dialogue_remote_call_and_playback_defer_announcements(self):
        node = self.node()
        node._dialogue_busy = True
        node._poll_startup_prompts()
        node._dialogue_busy = False
        node._remote_audio_gate.update('{"state":"start","mode":"session","epoch":7}')
        node._poll_startup_prompts()
        node._remote_audio_gate.close()
        node._player.foreground_active = True
        node._poll_startup_prompts()
        node._player.play_prompt.assert_not_called()
        node._player.foreground_active = False
        node._poll_startup_prompts()
        node._player.play_prompt.assert_called_once()

    def test_real_dialogue_interrupts_prompt_without_emitting_false_idle(self):
        node = self.node()
        node._poll_startup_prompts()
        beginning = node._startup_prompts.pending_request
        message = types.SimpleNamespace(data=b'\x03\x00' * 10,
                                        layout=types.SimpleNamespace(dim=[]))
        node._on_audio(message)
        node._player.stop_speech.assert_called_once()
        node._player.play.assert_called_once()
        np.testing.assert_array_equal(node._player.play.call_args.args[0], np.full(10, 3, dtype=np.int16))
        self.assertIsNone(node._startup_prompts.pending_request)
        self.assertTrue((self.root / 'startup_begin.finished').exists())
        node._dialog_state_pub.publish.assert_not_called()
        node._on_system_complete(beginning)  # Late callback cannot restart a gate.
        self.clock.return_value = 101.0
        self.assertFalse(StartupPromptGate().active())

    def test_ready_yields_to_active_utterance_and_asr_llm_handoff(self):
        node = self.node()
        node._poll_startup_prompts()
        node._on_system_complete(node._startup_prompts.pending_request)
        for component in self.required:
            mark_startup_ready(component)
        self.clock.return_value = 101.0
        node._on_startup_vad(types.SimpleNamespace(data=VAD_SPEECH_STARTED))
        node._on_startup_recognition_state(types.SimpleNamespace(data='busy'))
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        self.clock.return_value = 103.0
        node._on_startup_vad(types.SimpleNamespace(data=VAD_SPEECH_ENDED))
        self.clock.return_value = 200.0  # Valid ASR/fallback may exceed 30 or 90 seconds.
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        node._on_startup_recognition_state(types.SimpleNamespace(data='submitted'))
        self.clock.return_value = 300.0  # DDS handoff has not reached LLM yet.
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        node._on_dialogue_state(types.SimpleNamespace(data='busy'))
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        node._on_dialogue_state(types.SimpleNamespace(data='idle'))
        node._poll_startup_prompts()
        self.assertTrue(node._startup_prompts.pending_request.startswith('startup_ready-'))
        node._dialog_state_pub.publish.assert_not_called()

    def test_unknown_transcription_defers_optional_prompt_without_muting_recognition(self):
        node = self.node()
        node._on_startup_vad(types.SimpleNamespace(data=VAD_SPEECH_STARTED))
        self.clock.return_value = 180.0
        node._poll_startup_prompts()
        node._player.play_prompt.assert_not_called()
        self.clock.return_value = 10000.0
        node._poll_startup_prompts()
        node._player.play_prompt.assert_not_called()
        self.assertFalse(StartupPromptGate().active())

    def test_asr_failure_explicit_idle_releases_only_optional_cue_deferral(self):
        node = self.node()
        node._on_startup_vad(types.SimpleNamespace(data=VAD_SPEECH_STARTED))
        node._on_startup_recognition_state(types.SimpleNamespace(data='busy'))
        self.clock.return_value = 300.0
        node._poll_startup_prompts()
        node._player.play_prompt.assert_not_called()
        node._on_startup_recognition_state(types.SimpleNamespace(data='idle'))
        node._poll_startup_prompts()
        self.assertTrue(node._startup_prompts.pending_request.startswith('startup_begin-'))
        node._dialog_state_pub.publish.assert_not_called()

    def test_asr_submitted_after_llm_busy_does_not_leave_a_stale_handoff(self):
        node = self.node()
        node._poll_startup_prompts()
        node._on_system_complete(node._startup_prompts.pending_request)
        for component in self.required:
            mark_startup_ready(component)
        self.clock.return_value = 101.0
        node._on_startup_recognition_state(types.SimpleNamespace(data='busy'))
        node._on_dialogue_state(types.SimpleNamespace(data='busy'))
        node._on_startup_recognition_state(types.SimpleNamespace(data='submitted'))
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)
        node._on_dialogue_state(types.SimpleNamespace(data='idle'))
        node._poll_startup_prompts()
        self.assertTrue(node._startup_prompts.pending_request.startswith('startup_ready-'))

    def test_remote_audio_interrupts_prompt_and_keeps_realtime_epoch(self):
        node = self.node()
        node._poll_startup_prompts()
        node._remote_audio_gate.update('{"state":"start","mode":"session","epoch":8}')
        message = types.SimpleNamespace(data=b'\x04\x00' * 8, layout=types.SimpleNamespace(dim=[
            types.SimpleNamespace(label=REMOTE_AUDIO_EPOCH_LABEL_PREFIX + '8'),
        ]))
        node._on_remote_audio(message)
        self.assertIsNone(node._startup_prompts.pending_request)
        self.assertEqual(node._remote_audio_gate.epoch, 8)
        self.assertTrue(node._remote_audio_gate.active)
        node._player.play_realtime.assert_called_once()
        node._dialog_state_pub.publish.assert_not_called()

    def test_remote_disconnect_cancels_dropped_startup_completion(self):
        node = self.node()
        node._poll_startup_prompts()
        node._on_remote_intercom_state(types.SimpleNamespace(data='{"state":"start","mode":"session","epoch":9}'))
        node._on_remote_intercom_state(types.SimpleNamespace(data='{"state":"end","mode":"session","epoch":9}'))
        self.assertIsNone(node._startup_prompts.pending_request)
        self.assertTrue((self.root / 'startup_begin.finished').exists())
        self.clock.return_value = 101.0
        for component in self.required:
            mark_startup_ready(component)
        node._poll_startup_prompts()
        self.assertTrue(node._startup_prompts.pending_request.startswith('startup_ready-'))

    def test_shutdown_cancels_only_pending_startup_and_never_announces_ready(self):
        node = self.node()
        node._poll_startup_prompts()
        (self.root / 'stopping').touch()
        node._poll_startup_prompts()
        node._player.stop_speech.assert_called_once()
        self.assertIsNone(node._startup_prompts.pending_request)
        for component in self.required:
            mark_startup_ready(component)
        self.clock.return_value = 102.0
        node._poll_startup_prompts()
        self.assertEqual(node._player.play_prompt.call_count, 1)

    def test_standalone_playback_has_no_startup_timer_or_prompt(self):
        with patch.dict(os.environ, {'WALI_STARTUP_STATUS_DIR': ''}), \
             patch.object(self.module, 'MixingPlaybackService') as player:
            node = self.module.AudioPlaybackNode()
        self.assertEqual(node.timers, [])
        player.return_value.play_prompt.assert_not_called()


class StartupLauncherIntegrationTests(unittest.TestCase):
    def test_audio_is_spawned_first_while_other_order_and_shutdown_remain_intact(self):
        import launch_nodes
        names = ['camera_capture', 'llm', 'tts_play', 'audio_playback', 'stt']
        entries = [launch_nodes.NodeEntry(name, Path(name + '.py')) for name in names]
        started = []
        stopped = []

        def start(entry):
            started.append(entry.name)
            return types.SimpleNamespace(entry=entry, proc=types.SimpleNamespace(poll=lambda: None))

        def sleep(seconds):
            if seconds == 1.0:
                raise KeyboardInterrupt

        with patch.object(launch_nodes, 'build_node_list', return_value=entries), \
             patch.object(launch_nodes, 'validate_runtime_artifacts'), \
             patch.object(launch_nodes, 'start_process', side_effect=start), \
             patch.object(launch_nodes, 'stop_managed', side_effect=lambda managed: stopped.append(managed.entry.name)), \
             patch.object(launch_nodes.signal, 'signal'), \
             patch.object(launch_nodes.time, 'sleep', side_effect=sleep):
            with self.assertRaises(KeyboardInterrupt):
                launch_nodes.run_launcher(types.SimpleNamespace())
        self.assertEqual(started, ['audio_playback', 'camera_capture', 'llm', 'tts_play', 'stt'])
        self.assertEqual(stopped, list(reversed(started)))
        self.assertEqual(set(json.loads(entries[0].environment['WALI_STARTUP_REQUIRED'])),
                         {'audio_playback', 'tts_play', 'llm', 'stt', 'microphone'})
        self.assertFalse(Path(entries[0].environment['WALI_STARTUP_STATUS_DIR']).exists())
