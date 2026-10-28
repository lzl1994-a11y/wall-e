import unittest
from unittest.mock import patch

from services.llm import tool_dispatcher


class MultimodalToolIsolationTests(unittest.TestCase):
    def test_text_only_ordered_task_is_not_advertised_to_direct_audio(self):
        names = ['play_sequence', 'inspect_camera', 'search_environment', 'run_ordered_task']
        tools = [{'type': 'function', 'function': {'name': name}} for name in names]
        with patch.object(tool_dispatcher.mcp, 'get_chat_tools', return_value=tools):
            text_names = [tool['function']['name'] for tool in tool_dispatcher.get_tools()]
            audio_names = [tool['function']['name'] for tool in tool_dispatcher.get_multimodal_tools()]
        self.assertIn('run_ordered_task', text_names)
        self.assertNotIn('run_ordered_task', audio_names)
        self.assertEqual(audio_names, ['direct_answer', *names[:-1]])
        self.assertEqual([tool['function']['name'] for tool in tools], names)
