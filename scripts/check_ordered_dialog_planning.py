"""Check real model plans without loading ROS, cameras or motion transports.

Run explicitly: python scripts/check_ordered_dialog_planning.py --live
Only sanitized plans and verdicts are written to the report; never credentials.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.llm.llm_service import LLMService
from services.llm.llm_request_preparation import prepare_voice_request
from services.orchestration.ordered_task import (
    ORDERED_TASK_TOOL_NAME, ORDERED_TASK_PLANNING_POLICY, validate_ordered_task_plan,
)


CASES = [
    ('低头观察', '你低头，然后看看有什么。', ['head_down', 'inspect_camera']),
    ('连续连接词', '先低头，然后再看看前面有什么。', ['head_down', 'inspect_camera']),
    ('多动作并保存', '先低头，挥手，然后看看前面有什么，并拍照保存。', ['head_down', 'wave_hello', 'inspect_camera']),
    ('点头挥手观察', '先点头，再挥手，最后看看桌面上有什么。', ['basic_nod', 'wave_hello', 'inspect_camera']),
    ('转身找物', '先转身，然后找一下红色盒子。', ['move_chassis', 'search_environment']),
    ('转向搜索找到点头', '先向左转一秒，然后找一下红色盒子，找到后点头。', ['move_chassis', 'search_environment']),
    ('抬右手观察', '先抬起右手，然后看看前面有什么。', ['right_hand_up', 'inspect_camera']),
    ('观察后低头再看', '先看看前面有什么，然后低头再看一眼地面。', ['inspect_camera', 'head_down', 'inspect_camera']),
    ('翻译不执行', '把“低头然后看看”翻译成英文。', []),
    ('复述不执行', '请复述：先低头然后看看前面有什么。', []),
    ('能力询问不执行', '你会低头然后看看吗？', []),
]


def check(case, config_path):
    label, prompt, expected = case
    report = {'case': label, 'prompt': prompt, 'expected': expected}
    try:
        request = prepare_voice_request(prompt)
        llm = LLMService(config_path)
        llm.client = llm.client.with_options(timeout=45, max_retries=0)
        raw_plan = None
        calls = []
        text = prompt + '\n' + ORDERED_TASK_PLANNING_POLICY if expected else request.augmented_prompt
        for event in llm.chat_stream(
            text, [], tools_enabled=request.tools_enabled,
            only_action_name=ORDERED_TASK_TOOL_NAME if expected else None,
            max_tokens_override=1536 if expected else 256,
        ):
            if event.get('type') == 'tool_call':
                calls.append(event.get('name'))
                if event.get('name') == ORDERED_TASK_TOOL_NAME:
                    raw = event.get('arguments', {})
                    raw_plan = json.loads(raw) if isinstance(raw, str) else raw
        if not expected:
            report.update(passed=not calls, emitted_tools=calls)
            return report
        plan = validate_ordered_task_plan(prompt, raw_plan)
        actual = [s['arguments']['sequence_name'] if s['name'] == 'play_sequence' else s['name']
                  for s in plan['steps']]
        passed = actual == expected
        if label == '多动作并保存':
            passed = passed and plan['steps'][-1]['arguments'].get('save_photo') is True
        if label == '转向搜索找到点头':
            children = plan['steps'][-1]['arguments'].get('on_found_actions', [])
            passed = passed and any(c.get('arguments', {}).get('sequence_name') == 'basic_nod' for c in children)
        report.update(passed=bool(passed), actual=actual, plan=plan)
    except Exception as exc:
        # Provider exception messages may contain request details: keep only type.
        report.update(passed=False, error_type=type(exc).__name__)
        if isinstance(exc, ValueError) and str(exc).startswith('ordered_'):
            report['validation_error'] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Call the configured provider; never execute a plan')
    parser.add_argument('--output', default='reports/ordered-dialog-planning-20261029.json')
    parser.add_argument('--config', default=str(ROOT / 'core/config.yaml'))
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required to call the configured model')
    logging.disable(logging.CRITICAL)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(check, case, args.config) for case in CASES]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['case']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'mode': 'live_planning_only', 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{sum(r['passed'] for r in results)}/{len(results)} passed; report: {target}")
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
