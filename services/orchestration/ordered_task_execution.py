"""Execute a fully validated mixed plan without publishing partial dialog turns."""

from __future__ import annotations

from typing import Any, Callable


class OrderedTaskExecutor:
    def __init__(self, *, execute_motion: Callable, inspect: Callable,
                 search: Callable, cancelled: Callable, settle: Callable):
        self.execute_motion = execute_motion
        self.inspect = inspect
        self.search = search
        self.cancelled = cancelled
        self.settle = settle

    def invoke(self, plan: dict[str, Any]) -> dict[str, Any]:
        results = []
        answers = []
        steps = plan['steps']
        index = 0
        while index < len(steps):
            if self.cancelled():
                return self._outcome('interrupted', results, answers, 'turn_cancelled')
            step = steps[index]
            name = step['name']
            try:
                if name not in {'inspect_camera', 'search_environment'}:
                    batch = []
                    while index < len(steps) and steps[index]['name'] not in {'inspect_camera', 'search_environment'}:
                        batch.append(steps[index])
                        index += 1
                    state = self.execute_motion(batch)
                    records = state.get('results', [])
                    for offset, action in enumerate(batch):
                        record = records[offset] if offset < len(records) else {'status': 'failed', 'reason': 'missing_action_result'}
                        results.append({**action, **record})
                    if self.cancelled():
                        return self._outcome('interrupted', results, answers, 'turn_cancelled')
                    if state.get('status') != 'success' or len(records) != len(batch) or any(r.get('status') != 'completed' for r in records):
                        return self._outcome('failed', results, answers, 'ordered_action_failed')
                    if index < len(steps) and not self.settle():
                        return self._outcome('interrupted', results, answers, 'turn_cancelled')
                    continue
                callback = self.inspect if name == 'inspect_camera' else self.search
                result = callback(step['arguments'])
                results.append({**step, **result})
                if self.cancelled():
                    return self._outcome('interrupted', results, answers, 'turn_cancelled')
                if result.get('response'):
                    answers.append(result['response'])
                if result.get('status') != 'completed':
                    return self._outcome('failed', results, answers, result.get('reason', 'ordered_visual_step_failed'))
                # Conditional found-only actions remain inside the native
                # search.on_found_actions branch. Outer steps are unconditional.
                index += 1
            except Exception as exc:
                results.append({**step, 'status': 'failed', 'reason': type(exc).__name__})
                return self._outcome('failed', results, answers, 'ordered_step_failed')
        return self._outcome('completed', results, answers)

    @staticmethod
    def _outcome(status, results, answers, error=None):
        return {'status': status, 'results': results, 'response': ' '.join(answers), 'error': error}
