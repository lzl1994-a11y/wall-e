#!/usr/bin/env python3
"""Probe text replies and action routing without starting ROS or hardware.

The script reads the configured LLM credentials but never prints them.  It
only observes model output; returned action tool calls are not executed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.action_acknowledgement import action_acknowledgement
from services.action_intent_guard import validate_action_call
from services.llm_prompt import with_action_tool_policy, with_direct_speech_policy
from services.llm_request_options import reasoning_request_options
from services.llm_service import LLMService
from services.mcp_service import get_chat_tools
from services.tool_dispatcher import ToolCallAccumulator


@dataclass(frozen=True)
class Case:
    prompt: str
    expected_tool: str | None


CASES = {
    "chat": (
        Case("你好。", None),
        Case("你今天开心吗？", None),
        Case("你知道自己是谁吗？", None),
        Case("给我讲一个短笑话。", None),
        Case("今天的天气怎么样？", None),
        Case("你认识我吗？", None),
        Case("用一句话介绍你自己。", None),
        Case("为什么天空是蓝色的？", None),
        Case("一加一等于几？", None),
        Case("晚安。", None),
    ),
    "action": (
        Case("向左转头。", "play_sequence"),
        Case("看看右边。", "play_sequence"),
        Case("跟着我。", "set_tracking_mode"),
        Case("看着我。", "set_tracking_mode"),
        Case("停止跟随。", "set_tracking_mode"),
        Case("向前走两秒。", "move_chassis"),
        Case("后退一秒。", "move_chassis"),
        Case("做个开心的表情。", "express_emotion"),
        Case("向我招手。", "play_sequence"),
        Case("看看你前面有什么。", "inspect_camera"),
        Case("你能帮我向左转一下头吗？", "play_sequence"),
        Case("请跟着我好吗？", "set_tracking_mode"),
        Case("能不能向前走一下？", "move_chassis"),
        Case("别再跟着我了。", "set_tracking_mode"),
        Case("我不想让你跟着我。", "set_tracking_mode"),
    ),
    "boundary": (
        Case("你会走路吗？", None),
        Case("你能转头吗？", None),
        Case("你为什么看着我？", None),
        Case("跟着别人是什么意思？", None),
        Case("如果我让你前进，你会怎么做？", None),
        Case("给我讲一个关于机器人挥手的故事。", None),
        Case("你喜欢跳舞吗？", None),
        Case("你知道什么是挥手吗？", None),
        Case("我刚才向左看了。", None),
        Case("瓦力会跟随人吗？", None),
        Case("‘前进’这个词怎么读？", None),
        Case("我没有让你跟着我。", None),
        Case("不要向前走。", None),
        Case("别转头。", None),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "core" / "config.yaml")
    )
    parser.add_argument(
        "--suite", choices=(*CASES, "all"), default="all"
    )
    parser.add_argument(
        "--mode", choices=("standard", "envelope"), default="standard"
    )
    parser.add_argument(
        "--without-local-guard",
        action="store_true",
        help="Do not reject high-confidence non-command action proposals locally.",
    )
    return parser.parse_args()


def preview(text: str, limit: int = 72) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def request(client, settings, system_prompt, tools, prompt, *, tool_choice="auto"):
    kwargs = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings.get("temperature", 0.3),
        "max_tokens": min(int(settings.get("max_tokens", 256)), 256),
        "stream": True,
        "timeout": 30.0,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    kwargs.update(reasoning_request_options(settings))

    started = time.perf_counter()
    first_output = None
    finish_reason = ""
    content = []
    accumulator = ToolCallAccumulator()
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = str(choice.finish_reason)
        delta = choice.delta
        if delta.content:
            content.append(delta.content)
            first_output = first_output or time.perf_counter()
        if delta.tool_calls:
            first_output = first_output or time.perf_counter()
        accumulator.feed(delta)
    ended = time.perf_counter()
    return {
        "content": "".join(content).strip(),
        "calls": accumulator.flush(),
        "finish_reason": finish_reason,
        "first_latency": (first_output or ended) - started,
        "total_latency": ended - started,
    }


def request_service(service, prompt, *, tools_enabled=True):
    started = time.perf_counter()
    first_output = None
    content = []
    calls = []
    finish_reason = ""
    for event in service.chat_stream(prompt, tools_enabled=tools_enabled):
        event_type = event.get("type")
        if event_type == "text":
            content.append(event.get("content", ""))
            first_output = first_output or time.perf_counter()
        elif event_type == "tool_call":
            try:
                arguments = json.loads(event.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = None
            calls.append({"name": event.get("name"), "arguments": arguments})
            first_output = first_output or time.perf_counter()
        elif event_type == "done":
            finish_reason = event.get("finish_reason", "")
    ended = time.perf_counter()
    return {
        "content": "".join(content).strip(),
        "calls": calls,
        "finish_reason": finish_reason,
        "first_latency": (first_output or ended) - started,
        "total_latency": ended - started,
    }


def envelope_tool(action_tools):
    action_variants = []
    for tool in action_tools:
        function = tool["function"]
        action_variants.append({
            "type": "object",
            "description": function.get("description", ""),
            "properties": {
                "name": {"type": "string", "enum": [function["name"]]},
                "arguments": function["parameters"],
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return {
        "type": "function",
        "function": {
            "name": "robot_turn",
            "description": (
                "每轮必须调用一次的安全输出封套，本工具本身不执行动作。response 是唯一可播报"
                "台词；actions 只放用户明确要求瓦力现在执行的动作，没有明确动作命令时必须为空"
                "数组。不能为了让闲聊更生动而添加动作。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "简短自然、可直接播报的最终中文台词",
                    },
                    "actions": {
                        "type": "array",
                        "items": {"oneOf": action_variants},
                        "maxItems": 3,
                    },
                },
                "required": ["response", "actions"],
                "additionalProperties": False,
            },
        },
    }


def envelope_result(result):
    calls = [call for call in result["calls"] if call["name"] == "robot_turn"]
    if not calls:
        return {**result, "content": "", "calls": []}
    arguments = calls[-1]["arguments"]
    response = arguments.get("response")
    actions = arguments.get("actions")
    normalized = []
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict) or not isinstance(action.get("name"), str):
                continue
            normalized.append({
                "name": action["name"],
                "arguments": action.get("arguments", {}),
            })
    return {
        **result,
        "content": response.strip() if isinstance(response, str) else "",
        "calls": normalized,
    }


def retry_plain_text(service, prompt):
    retry_prompt = (
        f"用户说：{prompt}\n请直接简短回答用户原本的问题，不要执行或假装执行任何动作。"
    )
    return request_service(service, retry_prompt, tools_enabled=False)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    settings = config["llm"]
    service = LLMService(args.config)
    client = OpenAI(api_key=settings["key"], base_url=settings["url"])
    tools = get_chat_tools()
    system_prompt = with_action_tool_policy(
        with_direct_speech_policy(config.get("system_prompt", ""))
    )

    selected = CASES.values() if args.suite == "all" else (CASES[args.suite],)
    cases = [case for group in selected for case in group]
    print(
        f"provider={settings.get('provider')} model={settings.get('model')} mode={args.mode} "
        f"suite={args.suite} production_service={args.mode == 'standard'} "
        f"local_guard={not args.without_local_guard}"
    )

    passed = 0
    first_latencies = []
    total_latencies = []
    for index, case in enumerate(cases, 1):
        try:
            if args.mode == "standard":
                result = request_service(service, case.prompt)
            else:
                request_tools = [envelope_tool(tools)]
                tool_choice = {
                    "type": "function", "function": {"name": "robot_turn"}
                }
                result = request(
                    client,
                    settings,
                    system_prompt,
                    request_tools,
                    case.prompt,
                    tool_choice=tool_choice,
                )
                result = envelope_result(result)
        except Exception as exc:
            print(f"[{index:02d}] ERROR prompt={case.prompt!r} error={type(exc).__name__}")
            continue
        guarded = []
        reply_retry = False
        local_ack = False
        if result["calls"] and not args.without_local_guard:
            accepted = []
            for call in result["calls"]:
                allowed, reason = validate_action_call(
                    case.prompt,
                    call.get("name"),
                    call.get("arguments"),
                )
                if allowed:
                    accepted.append(call)
                else:
                    guarded.append(f"{call.get('name')}:{reason}")
            result["calls"] = accepted
        if guarded:
            if not result["content"]:
                retry = retry_plain_text(service, case.prompt)
                result["content"] = retry["content"]
                result["total_latency"] += retry["total_latency"]
                reply_retry = True
        if result["calls"] and not result["content"]:
            result["content"] = action_acknowledgement(result["calls"])
            local_ack = True
        names = [call["name"] for call in result["calls"]]
        actual = names[0] if names else None
        valid = actual == case.expected_tool
        if case.expected_tool is None:
            valid = valid and bool(result["content"])
        passed += int(valid)
        first_latencies.append(result["first_latency"])
        total_latencies.append(result["total_latency"])
        actual_label = actual or ("text" if result["content"] else "empty")
        print(
            f"[{index:02d}] {'PASS' if valid else 'FAIL'} expected={case.expected_tool or 'text'} "
            f"actual={actual_label} "
            f"first={result['first_latency']:.2f}s total={result['total_latency']:.2f}s "
            f"finish={result['finish_reason']!r} content={preview(result['content'])!r} "
            f"calls={json.dumps(names, ensure_ascii=False)} "
            f"guarded={json.dumps(guarded, ensure_ascii=False)} "
            f"reply_retry={reply_retry} local_ack={local_ack}"
        )

    total = len(cases)
    print(
        f"summary: passed={passed}/{total} first_p50={statistics.median(first_latencies):.2f}s "
        f"total_p50={statistics.median(total_latencies):.2f}s"
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
