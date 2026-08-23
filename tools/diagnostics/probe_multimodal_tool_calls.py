#!/usr/bin/env python3
"""Probe structured tool-call reliability without starting ROS or hardware.

The ``robot`` history mode intentionally mirrors the legacy broken
VoiceChatService behavior: it stores successful assistant replies without a
matching user transcript. Compare it with the default ``paired`` mode and
``none`` to diagnose structured-answer reliability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.llm_prompt import with_direct_speech_policy, with_structured_answer_policy
from services.llm_request_options import reasoning_request_options
from services.tool_dispatcher import (
    DIRECT_ANSWER_TOOL_NAME,
    MULTIMODAL_DIRECT_ANSWER_TOOL,
    ToolCallAccumulator,
    get_multimodal_tools,
)


DEFAULT_PROMPTS = (
    "拍照。",
    "你看前面有什么？",
    "你好。",
    "今天星期几？",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Probe direct_answer and camera tool calls against the configured LLM."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "core" / "config.yaml"),
        help="Project config path (API secrets are read but never printed).",
    )
    parser.add_argument(
        "--history-mode",
        choices=("robot", "paired", "none"),
        default="paired",
        help="robot mirrors the legacy bug; paired stores user+assistant; none is stateless.",
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        default=list(DEFAULT_PROMPTS),
        help="Prompts sent in order as one conversation.",
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Retry a missing direct_answer with only the forced answer tool.",
    )
    return parser.parse_args()


def _preview(value: str, limit: int = 160) -> str:
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[:limit] + "..."


def _request(
    client,
    settings,
    system_prompt,
    history,
    prompt,
    *,
    tools=None,
    tool_choice="auto",
):
    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": prompt})
    kwargs = {
        "model": settings["model"],
        "messages": messages,
        "modalities": ["text"],
        "tools": tools or get_multimodal_tools(),
        "tool_choice": tool_choice,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timeout": 15.0,
        "max_tokens": settings.get("max_tokens", 256),
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
    }
    kwargs.update(reasoning_request_options(settings))

    accumulator = ToolCallAccumulator()
    raw_content = []
    finish_reason = ""
    response = client.chat.completions.create(**kwargs)
    for chunk in response:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = str(choice.finish_reason)
        delta = choice.delta
        accumulator.feed(delta)
        if delta.content:
            raw_content.append(delta.content)

    calls = accumulator.flush()
    direct_answer = ""
    heard_text = ""
    for call in calls:
        if call["name"] != DIRECT_ANSWER_TOOL_NAME:
            continue
        candidate = call["arguments"].get("response")
        heard = call["arguments"].get("heard_text")
        if isinstance(heard, str):
            heard_text = heard.strip()
        if isinstance(candidate, str) and candidate.strip():
            direct_answer = candidate.strip()
    return {
        "finish_reason": finish_reason,
        "raw_content": "".join(raw_content).strip(),
        "calls": calls,
        "direct_answer": direct_answer,
        "heard_text": heard_text,
    }


def main():
    args = _parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    settings = config["llm"]
    client = OpenAI(api_key=settings["key"], base_url=settings["url"])
    system_prompt = with_structured_answer_policy(
        with_direct_speech_policy(config.get("system_prompt", ""))
    )

    print(
        f"provider={settings.get('provider')} model={settings.get('model')} "
        f"history_mode={args.history_mode}"
    )
    history = []
    failures = 0
    for index, prompt in enumerate(args.prompts, 1):
        roles = [message["role"] for message in history]
        print(f"\n[{index}] prompt={prompt!r} history_roles={roles}")
        try:
            result = _request(client, settings, system_prompt, history, prompt)
        except Exception as exc:
            failures += 1
            print(f"request_error={type(exc).__name__}: {_preview(exc)}")
            continue

        call_names = [call["name"] for call in result["calls"]]
        print(f"finish_reason={result['finish_reason']!r}")
        print(f"tool_calls={json.dumps(call_names, ensure_ascii=False)}")
        print(f"direct_answer={result['direct_answer']!r}")
        print(f"heard_text={result['heard_text']!r}")
        print(f"raw_content={_preview(result['raw_content'])!r}")
        if not result["direct_answer"]:
            print("diagnosis=MISSING_DIRECT_ANSWER")
            if args.retry_missing:
                try:
                    result = _request(
                        client,
                        settings,
                        system_prompt,
                        [],
                        prompt,
                        tools=[MULTIMODAL_DIRECT_ANSWER_TOOL],
                        tool_choice={
                            "type": "function",
                            "function": {"name": DIRECT_ANSWER_TOOL_NAME},
                        },
                    )
                except Exception as exc:
                    failures += 1
                    print(f"retry_error={type(exc).__name__}: {_preview(exc)}")
                    continue
                print(f"retry_direct_answer={result['direct_answer']!r}")
                print(f"retry_heard_text={result['heard_text']!r}")
            if not result["direct_answer"]:
                failures += 1
                continue

        if args.history_mode == "robot":
            history.append({"role": "assistant", "content": result["direct_answer"]})
        elif args.history_mode == "paired":
            history.extend((
                {"role": "user", "content": result["heard_text"] or prompt},
                {"role": "assistant", "content": result["direct_answer"]},
            ))

    print(f"\nsummary: turns={len(args.prompts)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
