"""Cross-boundary compound-dialog regressions without ROS or robot hardware.

Routing, plan authorization, node orchestration and existing workflows run for
real. Only model responses, camera frames, ROS transport and the native tree
owner are simulated. The native-owner fixture checks the submitted branch;
it does not claim to exercise the BehaviorTree.CPP runtime.
"""

import importlib
import json
import sys
import types
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from services.action.action_execution import CorrelatedActionExecutor
from services.action.action_status import build_action_status
from services.llm.llm_conditional_planning import (
    evaluate_conditional_plan,
    evaluate_native_conditional_event,
    parse_conditional_fallback_json,
)
from services.llm.llm_request_preparation import prepare_voice_request
from services.llm.llm_tool_proposal import evaluate_tool_proposal
from services.llm.voice_chat_service import VoiceChatService  # Preload before temporary ROS stubs.
from services.orchestration.ordered_task import validate_ordered_task_plan
from services.vision.visual_search import VisualSearchMotionSettings


def step(name, arguments, grounding):
    return {"name": name, "arguments": arguments, "grounding": grounding}


def pose(name, grounding):
    return step("play_sequence", {"sequence_name": name}, grounding)


def inspect(grounding, *, save=False):
    return step("inspect_camera", {"question": grounding, "save_photo": save}, grounding)


def conditional_plan():
    return {
        "observation": "观察前方有没有红盒子",
        "condition": "画面中找到红盒子",
        "action_name": "play_sequence",
        "action_arguments": {"sequence_name": "wave_hello"},
    }


def tool_event(name, arguments):
    return {
        "type": "tool_call", "name": name,
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


@pytest.fixture
def robot(tmp_path):
    class String:
        def __init__(self, data=""):
            self.data = data

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = object
    std_msgs = types.ModuleType("std_msgs.msg")
    std_msgs.String = std_msgs.UInt8MultiArray = String
    modules = {
        "rclpy": types.ModuleType("rclpy"), "rclpy.node": rclpy_node,
        "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": std_msgs,
    }
    module_name = "nodes.llm_ros_node"
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, modules):
        module = importlib.import_module(module_name)
    node = module.LLMBrainNode.__new__(module.LLMBrainNode)
    node.get_logger = MagicMock(return_value=MagicMock())
    node._action_executor = CorrelatedActionExecutor()
    node._game_mode, node._worker_running = "robot", True
    node._voice_task_cancelled = lambda: False
    node.chat_history = deque(maxlen=40)
    node.punctuations = {"。", "？", ".", "?", "！", "!"}
    node.llm = MagicMock()
    node.llm.settings = {}
    for field in (
        "tts_publisher", "corrected_publisher", "full_ai_publisher",
        "screen_dialog_publisher", "busy_publisher",
        "dialog_expression_publisher", "action_publisher",
    ):
        setattr(node, field, MagicMock())
    node.action_publisher.get_subscription_count.return_value = 1
    node.tft_preview_settings = types.SimpleNamespace(
        recognition_duration_ms=100, photo_directory=str(tmp_path),
    )
    frame = b"\xff\xd8fresh-compound-regression-frame\xff\xd9"
    events, commands = [], []

    def capture(**_kwargs):
        events.append("capture")
        return types.SimpleNamespace(last_frame=frame, busy=False, error=None)

    def analyze(image, _question):
        assert image == frame
        events.append("analyze")
        return "前面有一个红盒子。"

    def publish(message):
        command = json.loads(message.data)
        if isinstance(command.get("arguments"), str):
            command["arguments"] = json.loads(command["arguments"])
        commands.append(command)
        events.append(command["arguments"].get("sequence_name", command["name"]))
        if command.get("request_id"):
            node._action_executor.accept_status(build_action_status(
                command["request_id"], command["name"], "completed",
                source="compound_regression",
            ))

    node.action_publisher.publish.side_effect = publish
    node._run_camera_preview = MagicMock(side_effect=capture)
    node._analyze_camera_frame = MagicMock(side_effect=analyze)
    rig = types.SimpleNamespace(
        node=node, module=module, events=events, commands=commands,
        photo_directory=tmp_path,
    )
    try:
        yield rig
    finally:
        sys.modules.pop(module_name, None)


def run_dialog(robot, prompt, arguments, *, tool="run_ordered_task"):
    robot.node.llm.chat_stream.return_value = iter([tool_event(tool, arguments)])
    robot.node._process_voice_task("compound-regression", prompt)


@pytest.mark.parametrize("suffix", ["找到后挥手", "如果找到就挥手"])
@pytest.mark.parametrize("search_status", ["found", "not_found", "uncertain"])
def test_complete_search_request_preserves_head_and_found_only_branch(robot, suffix, search_status):
    prompt = f"先低头，然后找红盒子，{suffix}"
    assert prepare_voice_request(prompt).route == "ordered_task"
    submitted = []

    def native_owner(plan, **_kwargs):
        # This is the transport boundary: only the owner executes these nested
        # actions. The dialogue node must not publish an unconditional wave.
        submitted.append(plan)
        assert plan.root_type == "VisualSearchThen"
        assert len(plan.on_found_actions) == 1
        assert plan.on_found_actions[0]["arguments"] == {"sequence_name": "wave_hello"}
        robot.events.append(f"search:{search_status}")
        records = [{
            "action": "search_environment", "status": "completed",
            "found": search_status == "found", "response": f"搜索结果：{search_status}",
        }]
        if search_status == "found":
            robot.events.append("native:wave_hello")
            records.append({"action": "play_sequence", "status": "completed"})
        return {"status": "success", "results": records}

    robot.node._try_execute_native_plan = MagicMock(side_effect=native_owner)
    with patch.object(robot.module, "load_visual_search_motion_settings",
                      return_value=VisualSearchMotionSettings()):
        run_dialog(robot, prompt, {"steps": [
            pose("head_down", "低头"),
            step("search_environment", {
                "target": "红盒子", "on_found_actions": [{
                    "name": "play_sequence", "arguments": {"sequence_name": "wave_hello"},
                }],
            }, f"找红盒子，{suffix}"),
        ]})
    expected = ["head_down", f"search:{search_status}"]
    if search_status == "found":
        expected.append("native:wave_hello")
    assert robot.events == expected
    assert len(submitted) == 1
    assert [command["arguments"].get("sequence_name") for command in robot.commands] == ["head_down"]


@pytest.mark.parametrize("suffix", ["找到后挥手", "如果找到就挥手"])
def test_incomplete_conditional_plan_is_rejected_at_every_entry(suffix):
    prompt = f"先低头，然后找红盒子，{suffix}"
    plan = conditional_plan()
    event = tool_event("run_conditional_task", plan)
    assert prepare_voice_request(prompt).route == "ordered_task"
    for conditional_request in (False, True):
        assert evaluate_tool_proposal(
            event, user_prompt=prompt, conditional_request=conditional_request,
        ).is_rejected
    assert evaluate_native_conditional_event(event, user_prompt=prompt).is_rejected
    parsed = parse_conditional_fallback_json(json.dumps(plan, ensure_ascii=False))
    assert evaluate_conditional_plan(parsed, user_prompt=prompt).is_rejected


@pytest.mark.parametrize("response_kind", ["native", "json"])
def test_node_rejects_a_conditional_response_that_drops_the_prerequisite(robot, response_kind):
    prompt = "先低头，然后找红盒子，如果找到就挥手"
    plan = conditional_plan()
    event = (tool_event("run_conditional_task", plan) if response_kind == "native"
             else {"type": "text", "content": json.dumps(plan, ensure_ascii=False)})
    robot.node.llm.chat_stream.side_effect = lambda *_args, **_kwargs: iter([event])
    robot.node._try_execute_native_plan = MagicMock()
    robot.node._process_voice_task("compound-regression", prompt)
    assert robot.events == []
    robot.node._try_execute_native_plan.assert_not_called()
    assert list(robot.photo_directory.iterdir()) == []


def test_stopping_music_does_not_reclassify_the_following_inspection(robot):
    prompt = "先停止音乐，然后看看前面有什么"
    run_dialog(robot, prompt, {"steps": [
        step("control_music", {"action": "stop"}, "停止音乐"),
        inspect("看看前面有什么"),
    ]})
    assert robot.events == ["control_music", "capture", "analyze"]
    assert robot.commands[0]["arguments"] == {"action": "stop"}


def test_no_photo_constraint_allows_head_and_inspection_without_saving(robot):
    prompt = "先低头，然后看看前面有什么，不要拍照"
    run_dialog(robot, prompt, {"steps": [
        pose("head_down", "低头"), inspect("看看前面有什么"),
    ]})
    assert robot.events == ["head_down", "capture", "analyze"]
    assert list(robot.photo_directory.iterdir()) == []


def test_asr_without_punctuation_preserves_the_local_photo_prohibition(robot):
    prompt = "先低头看看不要拍照"
    run_dialog(robot, prompt, {"steps": [
        pose("head_down", "低头"), inspect("看看不要拍照"),
    ]})
    assert robot.events == ["head_down", "capture", "analyze"]
    assert list(robot.photo_directory.iterdir()) == []


def test_no_photo_constraint_rejects_save_before_any_effect(robot):
    prompt = "先低头，然后看看前面有什么，不要拍照"
    run_dialog(robot, prompt, {"steps": [
        pose("head_down", "低头"), inspect("看看前面有什么", save=True),
    ]})
    assert robot.events == []
    assert list(robot.photo_directory.iterdir()) == []


def test_forbidden_head_motion_is_not_a_required_step(robot):
    prompt = "不要低头，直接看看前面有什么"
    run_dialog(robot, prompt, {"steps": [inspect("看看前面有什么")]})
    assert robot.events == ["capture", "analyze"]
    assert robot.commands == []


@pytest.mark.parametrize("prompt, observation", [
    ("不要低头然后挥手再看", "看"),
    ("不要低头，然后挥手，再看看前面有什么", "看看前面有什么"),
])
def test_local_negation_does_not_forbid_the_following_wave(robot, prompt, observation):
    run_dialog(robot, prompt, {"steps": [pose("wave_hello", "挥手"), inspect(observation)]})
    assert robot.events == ["wave_hello", "capture", "analyze"]
    assert all(command["arguments"].get("sequence_name") != "head_down" for command in robot.commands)


def test_truncated_grounding_cannot_hide_original_negation(robot):
    prompt = "不要低头，然后看看前面有什么"
    plan = {"steps": [pose("head_down", "低头"), inspect("看看前面有什么")]}
    with pytest.raises(ValueError):
        validate_ordered_task_plan(prompt, plan)
    assert evaluate_tool_proposal(tool_event("run_ordered_task", plan), user_prompt=prompt).is_rejected
    run_dialog(robot, prompt, plan)
    assert robot.events == []


@pytest.mark.parametrize("with_wave", [False, True])
def test_wide_model_evidence_preserves_local_negation_through_execution(robot, with_wave):
    if with_wave:
        prompt = "不要低头，然后挥手，再看看前面有什么"
        steps = [pose("wave_hello", "不要低头，然后挥手"), inspect("看看前面有什么")]
    else:
        prompt = "不要低头，直接看看前面有什么"
        steps = [inspect(prompt)]
    run_dialog(robot, prompt, {"steps": steps})
    assert robot.events == (["wave_hello", "capture", "analyze"] if with_wave
                            else ["capture", "analyze"])
    assert all(command["arguments"].get("sequence_name") != "head_down" for command in robot.commands)


@pytest.mark.parametrize("repair_valid", [False, True])
def test_repair_validates_the_whole_plan_before_any_action(robot, repair_valid):
    prompt = "不要低头，然后挥手，再看看前面有什么"
    valid = {"steps": [pose("wave_hello", "挥手"), inspect("看看前面有什么")]}
    invalid = {"steps": [pose("head_down", "不要低头"), *valid["steps"]]}
    calls = []

    def model(text, *_args, **_kwargs):
        calls.append(text)
        assert robot.events == []
        assert text.startswith(prompt)
        candidate = valid if repair_valid and len(calls) == 2 else invalid
        return iter([tool_event("run_ordered_task", candidate)])

    robot.node.llm.chat_stream.side_effect = model
    robot.node._process_voice_task("compound-repair", prompt)
    assert len(calls) == 2
    assert "ordered_plan_negated_action" in calls[1]
    assert robot.events == (["wave_hello", "capture", "analyze"] if repair_valid else [])
    assert all(command["arguments"].get("sequence_name") != "head_down" for command in robot.commands)


@pytest.mark.parametrize("prompt", [
    "把‘低头然后看看’翻译成英文",
    "解释一下‘低头然后看看’这句话是什么意思",
])
def test_translation_and_quotation_do_not_execute_an_injected_plan(robot, prompt):
    assert prepare_voice_request(prompt).tools_enabled is False
    run_dialog(robot, prompt, {"steps": [pose("head_down", "低头"), inspect("看看")]})
    assert robot.events == []


@pytest.mark.parametrize("decision", ["yes", "no", "uncertain"])
def test_existing_pure_conditional_workflow_remains_available(robot, decision):
    prompt = "如果看到红盒子就挥手"
    plan = conditional_plan()
    assert prepare_voice_request(prompt).route == "conditional_task"
    assert evaluate_native_conditional_event(
        tool_event("run_conditional_task", plan), user_prompt=prompt,
    ).is_accepted
    assert evaluate_conditional_plan(
        parse_conditional_fallback_json(json.dumps(plan, ensure_ascii=False)), user_prompt=prompt,
    ).is_accepted
    robot.node._evaluate_camera_condition = MagicMock(return_value={
        "decision": decision, "evidence": "模拟视觉结果",
    })
    run_dialog(robot, prompt, plan, tool="run_conditional_task")
    assert robot.events == (["capture", "wave_hello"] if decision == "yes" else ["capture"])
