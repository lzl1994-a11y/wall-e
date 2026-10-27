import unittest

from services.llm.llm_request_preparation import prepare_voice_request
from services.orchestration.action_before_inspection import action_before_inspection


class ActionBeforeInspectionTests(unittest.TestCase):
    def test_examples_route_to_grounded_head_down(self):
        for text in ('你低头，然后看看有什么。', '你必须先低头，然后才能看。',
                     '你低头，然后看看前面有什么。'):
            with self.subTest(text=text):
                request = prepare_voice_request(text)
                self.assertEqual(request.route, 'action_before_inspection')
                self.assertEqual(request.pre_inspection_actions[0]['arguments'],
                                 {'sequence_name': 'head_down'})

    def test_multiple_explicit_steps_preserve_order(self):
        actions = action_before_inspection('先低头再挥手然后看看')
        self.assertEqual([a['arguments']['sequence_name'] for a in actions],
                         ['head_down', 'wave_hello'])

    def test_noncommands_and_ambiguous_combination_do_not_form_a_plan(self):
        for text in ('你会低头然后看看吗？', '不要低头然后看看', '他低头然后看看',
                     '先低头并挥手然后看看', '低头然后看电影'):
            with self.subTest(text=text):
                self.assertIsNone(action_before_inspection(text))
