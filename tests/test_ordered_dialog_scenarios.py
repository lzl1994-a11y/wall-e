"""Exercise mixed dialog plans through real validation and correlated actions.

The LLM transport and peripheral camera/native-search endpoints are simulated;
the tests do not contact a provider or move robot hardware.
"""

import importlib
import json
import sys
import tempfile
import threading
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.action.action_execution import CorrelatedActionExecutor
from services.action.action_status import build_action_status
from services.llm.voice_chat_service import VoiceChatService  # Preload imports before temporary ROS stubs.
from services.vision.visual_search import VisualSearchMotionSettings


def step(name, arguments, grounding):
    return {'name': name, 'arguments': arguments, 'grounding': grounding}


def pose(name, grounding):
    return step('play_sequence', {'sequence_name': name}, grounding)


def inspect(grounding, *, save=False):
    return step('inspect_camera', {'question': grounding, 'save_photo': save}, grounding)


class OrderedDialogScenarioTests(unittest.TestCase):
    def _node(self):
        class String:
            def __init__(self, data=''):
                self.data = data

        rclpy_node = types.ModuleType('rclpy.node')
        rclpy_node.Node = object
        std_msgs = types.ModuleType('std_msgs.msg')
        std_msgs.String = std_msgs.UInt8MultiArray = String
        modules = {
            'rclpy': types.ModuleType('rclpy'), 'rclpy.node': rclpy_node,
            'std_msgs': types.ModuleType('std_msgs'), 'std_msgs.msg': std_msgs,
        }
        module_name = 'nodes.llm_ros_node'
        sys.modules.pop(module_name, None)
        with patch.dict(sys.modules, modules):
            module = importlib.import_module(module_name)
        self.addCleanup(sys.modules.pop, module_name, None)
        node = module.LLMBrainNode.__new__(module.LLMBrainNode)
        node.get_logger = MagicMock(return_value=MagicMock())
        node._action_executor = CorrelatedActionExecutor()
        node._game_mode, node._worker_running = 'robot', True
        node.chat_history = deque(maxlen=40)
        node.punctuations = {'。', '？', '.', '?', '！', '!'}
        node.llm = MagicMock()
        node.llm.settings = {}
        self.cancelled = threading.Event()
        node._voice_task_cancelled = self.cancelled.is_set
        for field in ('tts_publisher', 'corrected_publisher', 'full_ai_publisher',
                      'screen_dialog_publisher', 'busy_publisher',
                      'dialog_expression_publisher', 'action_publisher'):
            setattr(node, field, MagicMock())
        node.action_publisher.get_subscription_count.return_value = 1
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        node.tft_preview_settings = types.SimpleNamespace(
            recognition_duration_ms=100, photo_directory=folder.name,
        )
        self.events = []
        self.frame = b'\xff\xd8fresh-frame-after-motion\xff\xd9'

        def capture(**_kwargs):
            self.events.append('capture')
            return types.SimpleNamespace(last_frame=self.frame, busy=False, error=None)

        def analyze(frame, _question):
            self.assertEqual(frame, self.frame)
            self.events.append('analyze')
            return '地面上有一个红色盒子。'

        def publish(message):
            command = json.loads(message.data)
            self.events.append(command['arguments'].get('sequence_name', command['name']))
            self._status(node, command, 'completed')

        node.action_publisher.publish.side_effect = publish
        node._run_camera_preview = MagicMock(side_effect=capture)
        node._analyze_camera_frame = MagicMock(side_effect=analyze)
        return node, module, Path(folder.name)

    @staticmethod
    def _status(node, command, status):
        node._action_executor.accept_status(build_action_status(
            command['request_id'], command['name'], status, source='scenario_test',
        ))

    def _dialog(self, node, prompt, steps, *, tool_name='run_ordered_task'):
        arguments = {'steps': steps} if tool_name == 'run_ordered_task' else steps
        node.llm.chat_stream.return_value = iter([{
            'type': 'tool_call', 'name': tool_name,
            'arguments': json.dumps(arguments, ensure_ascii=False),
        }])
        node._process_voice_task('scenario-turn', prompt)

    def test_head_down_then_new_camera_frame_across_natural_phrasings(self):
        scenarios = (
            ('你低头，然后看看有什么。', '看看有什么'),
            ('你必须先低头，然后才能看。', '才能看'),
            ('先低头，然后再看看前面有什么。', '看看前面有什么'),
            ('先低头，然后看看前面有什么，再告诉我。', '看看前面有什么'),
        )
        for prompt, question in scenarios:
            with self.subTest(prompt=prompt):
                node, _, folder = self._node()
                self._dialog(node, prompt, [pose('head_down', '低头'), inspect(question)])
                self.assertEqual(self.events, ['head_down', 'capture', 'analyze'])
                self.assertEqual(list(folder.iterdir()), [])

    def test_comma_separated_actions_are_not_dropped_and_photo_is_saved(self):
        node, _, folder = self._node()
        self._dialog(node, '先低头，挥手，然后看看前面有什么，并拍照保存。', [
            pose('head_down', '低头'), pose('wave_hello', '挥手'),
            inspect('看看前面有什么，并拍照保存', save=True),
        ])
        self.assertEqual(self.events[:2], ['head_down', 'wave_hello'])
        self.assertEqual(self.events.count('capture'), 1)
        self.assertEqual(self.events.count('analyze'), 1)
        photos = list(folder.glob('*.jpg'))
        self.assertEqual(len(photos), 1)
        self.assertEqual(photos[0].read_bytes(), self.frame)

    def test_nod_wave_then_inspect_retains_each_motion(self):
        node, _, _ = self._node()
        self._dialog(node, '先点头，再挥手，最后看看桌上有什么。', [
            pose('basic_nod', '点头'), pose('wave_hello', '挥手'),
            inspect('看看桌上有什么'),
        ])
        self.assertEqual(self.events, ['basic_nod', 'wave_hello', 'capture', 'analyze'])

    def test_bounded_turn_precedes_visual_search(self):
        node, _, _ = self._node()

        def search(arguments, turn_id):
            self.events.append('search')
            self.assertEqual(arguments['target'], '红色盒子')
            return {'status': 'completed', 'found': True, 'response': '红色盒子在后面。'}

        node._execute_ordered_search_step = MagicMock(side_effect=search)
        self._dialog(node, '先原地转圈一秒，然后寻找红色盒子。', [
            step('move_chassis', {'direction': 'spin', 'duration': 1}, '原地转圈一秒'),
            step('search_environment', {'target': '红色盒子'}, '寻找红色盒子'),
        ])
        self.assertEqual(self.events, ['move_chassis', 'search'])
        node._execute_ordered_search_step.assert_called_once()

    def test_found_only_followup_remains_owned_by_native_search(self):
        node, module, _ = self._node()
        plans = []

        def native_search(plan, **_kwargs):
            plans.append(plan)
            return {'status': 'success', 'results': [{
                'action': 'search_environment', 'status': 'completed',
                'found': False, 'response': '这次没找到红色盒子。',
            }]}

        node._try_execute_native_plan = MagicMock(side_effect=native_search)
        with patch.object(module, 'load_visual_search_motion_settings',
                          return_value=VisualSearchMotionSettings()):
            self._dialog(node, '先低头，然后寻找红色盒子，找到后点头。', [
                pose('head_down', '低头'),
                step('search_environment', {
                    'target': '红色盒子',
                    'on_found_actions': [{
                        'name': 'play_sequence', 'arguments': {'sequence_name': 'basic_nod'},
                    }],
                }, '寻找红色盒子，找到后点头'),
            ])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].on_found_actions[0]['arguments'],
                         {'sequence_name': 'basic_nod'})
        self.assertEqual(self.events, ['head_down'])

    def test_motion_accepted_is_not_enough_to_capture(self):
        node, _, _ = self._node()
        started = threading.Event()
        commands = []

        def publish(message):
            command = json.loads(message.data)
            commands.append(command)
            self._status(node, command, 'accepted')
            started.set()

        node.action_publisher.publish.side_effect = publish
        worker = threading.Thread(target=lambda: self._dialog(
            node, '先低头，然后看看有什么。',
            [pose('head_down', '低头'), inspect('看看有什么')]), daemon=True)
        worker.start()
        try:
            self.assertTrue(started.wait(3), 'motion was never submitted')
            node._run_camera_preview.assert_not_called()
            self._status(node, commands[0], 'completed')
            worker.join(4)
            self.assertFalse(worker.is_alive())
            node._run_camera_preview.assert_called_once()
        finally:
            self.cancelled.set()
            worker.join(2)

    def test_failed_motion_stops_later_motion_and_camera(self):
        node, _, _ = self._node()

        def publish(message):
            command = json.loads(message.data)
            self.events.append(command['arguments']['sequence_name'])
            self._status(node, command, 'failed')

        node.action_publisher.publish.side_effect = publish
        self._dialog(node, '先低头，再挥手，然后看看有什么。', [
            pose('head_down', '低头'), pose('wave_hello', '挥手'), inspect('看看有什么'),
        ])
        self.assertNotIn('wave_hello', self.events)
        node._run_camera_preview.assert_not_called()

    def test_timeout_does_not_capture(self):
        node, _, _ = self._node()
        node._execute_dialog_action_sequence = MagicMock(return_value={
            'status': 'failure', 'results': [{
                'name': 'play_sequence', 'arguments': {'sequence_name': 'head_down'},
                'status': 'timeout', 'reason': 'action_timeout',
            }],
        })
        self._dialog(node, '先低头，然后看看有什么。',
                     [pose('head_down', '低头'), inspect('看看有什么')])
        node._run_camera_preview.assert_not_called()

    def test_cancel_after_motion_prevents_camera(self):
        node, _, _ = self._node()

        def publish(message):
            command = json.loads(message.data)
            self._status(node, command, 'completed')
            self.cancelled.set()

        node.action_publisher.publish.side_effect = publish
        self._dialog(node, '先低头，然后看看有什么。',
                     [pose('head_down', '低头'), inspect('看看有什么')])
        node._run_camera_preview.assert_not_called()

    def test_invalid_last_step_rejects_whole_plan_before_first_motion(self):
        node, _, _ = self._node()
        self._dialog(node, '先低头，然后看看有什么。', [
            pose('head_down', '低头'),
            step('inspect_camera', {'save_photo': 'yes'}, '看看有什么'),
        ])
        node.action_publisher.publish.assert_not_called()
        node._run_camera_preview.assert_not_called()

    def test_model_cannot_silently_drop_named_head_motion(self):
        node, _, _ = self._node()
        self._dialog(node, '先低头，挥手，然后看看前面有什么。', [
            pose('wave_hello', '挥手'), inspect('看看前面有什么'),
        ])
        node.action_publisher.publish.assert_not_called()
        node._run_camera_preview.assert_not_called()

    def test_model_cannot_silently_drop_requested_photo_save(self):
        node, _, folder = self._node()
        self._dialog(node, '先低头，然后看看前面有什么，并拍照保存。', [
            pose('head_down', '低头'), inspect('看看前面有什么', save=False),
        ])
        node.action_publisher.publish.assert_not_called()
        node._run_camera_preview.assert_not_called()
        self.assertEqual(list(folder.iterdir()), [])

    def test_translation_and_repetition_cannot_execute_even_with_bad_model_proposal(self):
        for prompt in ('把“低头然后看看”翻译成英文。', '请复述：低头然后看看。',
                       '请读一下：低头然后看看。'):
            with self.subTest(prompt=prompt):
                node, _, _ = self._node()
                self._dialog(node, prompt, [pose('head_down', '低头'), inspect('看看')])
                node.action_publisher.publish.assert_not_called()
                node._run_camera_preview.assert_not_called()

    def test_single_camera_tool_cannot_drop_prerequisite_motion(self):
        node, _, _ = self._node()
        self._dialog(node, '先低头，然后看看前面有什么。',
                     {'question': '前面有什么'}, tool_name='inspect_camera')
        node._run_camera_preview.assert_not_called()
        node.action_publisher.publish.assert_not_called()

    def test_camera_failure_records_already_completed_motion(self):
        node, _, _ = self._node()
        node._run_camera_preview.return_value = types.SimpleNamespace(
            last_frame=None, busy=False, error='camera_frame_unavailable',
        )
        node._run_camera_preview.side_effect = None
        self._dialog(node, '先低头，然后看看有什么。',
                     [pose('head_down', '低头'), inspect('看看有什么')])
        motion_results = [json.loads(item['content']) for item in node.chat_history
                          if item.get('role') == 'tool' and item.get('name') == 'play_sequence']
        self.assertTrue(any(result.get('status') == 'completed' for result in motion_results))
        node._analyze_camera_frame.assert_not_called()
