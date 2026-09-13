"""直接语音对话服务：AudioPipeline(采集+唤醒词+VAD) → state machine → multimodal LLM → TTS

状态机:
  IDLE        ── 待机，等待唤醒词
  AWAKE       ── 已唤醒，VAD 监听语音
  LLM_PENDING ── 语音已发送，等待 LLM 回复
  SPEAKING    ── 等待扬声器播放完成，麦克风保持暂停

唤醒词触发 → 播放预合成语音 → TFT 切聊天页 → 进入 AWAKE
LLM 40s 无回复 → 超时回到 IDLE，需重新唤醒
LLM 交互中听到唤醒词 → 强制中断 LLM → 播放语音应答 → AWAKE
"""

import base64
import os
import tempfile
import threading
import time
import wave
import yaml
from collections import deque
from enum import Enum, auto

from openai import OpenAI
from services.llm_prompt import (
    with_action_tool_policy,
    with_dialog_expression_policy,
    with_direct_speech_policy,
    with_structured_answer_policy,
)
from services.llm_request_options import normalize_tool_choice, reasoning_request_options
from services.tool_dispatcher import (
    DIRECT_ANSWER_TOOL,
    DIRECT_ANSWER_TOOL_NAME,
    MULTIMODAL_DIRECT_ANSWER_TOOL,
    ToolCallAccumulator,
    build_action_cmd,
    get_action_tools,
    get_multimodal_tools,
)
from services.dialog_expression_protocol import normalize_expression
from services.conditional_task import (
    CONDITIONAL_DECISION_TOOL_NAME,
    CONDITIONAL_TASK_TOOL_NAME,
    conditional_decision_tool,
    parse_conditional_decision,
)
from services.action_intent_guard import validate_action_arguments
from services.behavior_tree_workflow import BehaviorTreeActionWorkflow
from services.visual_search import (
    VISUAL_SEARCH_RESULT_TOOL_NAME,
    VISUAL_SEARCH_TOOL_NAME,
    parse_visual_search_result,
    visual_search_result_tool,
)
from .audio_pipeline import AudioPipeline
from .multimodal import create_multimodal
from .voice_debug import RollingVoiceDebugStore


class _State(Enum):
    IDLE = auto()
    AWAKE = auto()
    LLM_PENDING = auto()
    SPEAKING = auto()


class VoiceChatService:
    """直接语音对话服务（唤醒词版）。

    Usage:
        vc = VoiceChatService(config_path="core/config.yaml")
        vc.on_wake_word    = your_wake_handler
        vc.on_llm_reply    = lambda text: your_tts(text)
        vc.on_llm_chunk    = lambda text: stream_tts(text)
        vc.on_tool_call    = lambda name, args: your_action(name, args)
        vc.on_llm_done     = your_turn_done_handler
        vc.on_llm_timeout  = your_timeout_handler
        vc.start()
    """

    SAMPLE_RATE = AudioPipeline.SAMPLE_RATE
    API_TIMEOUT = 10.0
    LLM_IDLE_TIMEOUT = 40.0
    FALLBACK_REPLY = "这次我没听清，请再说一遍。"

    def __init__(self, config_path="core/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_cfg = config["llm"]
        self.client = OpenAI(api_key=llm_cfg["key"], base_url=llm_cfg["url"])
        self.multimodal = create_multimodal(config_path)
        self._voice_debug = RollingVoiceDebugStore()
        self.model = llm_cfg["model"]
        self.max_tokens = llm_cfg.get("max_tokens", 1024)
        self.llm_settings = llm_cfg
        self.system_prompt = with_dialog_expression_policy(
            with_action_tool_policy(
                with_direct_speech_policy(config.get("system_prompt", ""))
            )
        )

        # 对话历史（最近20轮）
        self._chat_history: deque = deque(maxlen=40)

        # 唤醒应答 WAV
        ww_cfg = config.get("wake_word", {})
        self._wake_response_wav = ww_cfg.get("response_wav", "assets/wake_response.wav")

        # ── AudioPipeline：统一处理采集+唤醒词+VAD断句 ──
        self._pipe = AudioPipeline(config_path)
        self._pipe.on_wake_word = self._on_wake_detected
        self._pipe.on_sentence = self._on_sentence
        self._pipe.on_speech_start = self._on_speech_start
        self._pipe.on_speech_cancel = self._on_speech_cancel

        # ── 状态机 ──
        self._state = _State.IDLE
        self._state_lock = threading.Lock()
        self._last_llm_activity = 0.0
        self._cancel_llm = threading.Event()
        self._llm_thread = None

        # ── 回调 ──
        self.on_wake_word = None       # 唤醒词触发（应播放应答语音、切 TFT 页面）
        self.on_speech_start = None    # VAD 首帧（仅已唤醒状态）
        self.on_speech_end = None      # VAD 断句或取消（仅已唤醒状态）
        self.on_llm_reply = None       # LLM 文本回复（最终完整回复）
        self.on_llm_chunk = None       # LLM 流式文本块
        self.on_expression = None      # LLM 语义表情 (expression, intensity)
        self.on_tool_call = None       # LLM 工具调用
        self.on_action_plan = None     # 原生行为树计划提交（不可用时返回 None）
        self.on_photo_request = None   # 多模态拍照请求（由 ROS 节点执行）
        self.on_inspection_request = None  # 多模态看图请求（由 ROS 节点执行）
        self.on_llm_done = None        # LLM 本轮结束（成功、失败或取消）
        self.on_llm_timeout = None     # 40s 无回复超时

    # ================================================================
    # Public API
    # ================================================================
    def start(self):
        self._pipe.start()
        self._last_llm_activity = time.time()

        # 超时监控线程
        threading.Thread(target=self._timeout_watch, daemon=True).start()

        print(f"[VoiceChat] 已启动 (直接语音对话)")

    def stop(self):
        self._cancel_llm.set()
        self._pipe.stop()
        if self._llm_thread and self._llm_thread.is_alive():
            self._llm_thread.join(timeout=3.0)
        print("[VoiceChat] 已停止")

    def pause(self):
        self._pipe.pause()
        self._cancel_llm.set()
        with self._state_lock:
            self._state = _State.IDLE
        print("[VoiceChat] 已暂停")

    def resume(self):
        self._pipe.resume()
        print("[VoiceChat] 已恢复")

    def begin_output_playback(self):
        """Mute capture while robot audio is playing through the speaker."""
        with self._state_lock:
            self._state = _State.SPEAKING
        self._pipe.pause()
        print("[VoiceChat] 播放期间暂停麦克风")

    def complete_output_playback(self):
        """Resume capture after playback and the acoustic echo tail are over."""
        with self._state_lock:
            if self._state != _State.SPEAKING:
                return False
            self._state = _State.AWAKE
        self._last_llm_activity = time.time()
        self._pipe.resume()
        print("[VoiceChat] 播放完成，恢复麦克风")
        return True

    # ================================================================
    # 状态机入口
    # ================================================================
    def _on_wake_detected(self):
        """唤醒词触发：AudioPipeline 回调（已在音频线程内）。"""
        now = time.time()

        # 如果 LLM 正在跑，中断它
        if self._llm_thread and self._llm_thread.is_alive():
            print("[VoiceChat] 强制中断当前 LLM 对话")
            self._cancel_llm.set()
            self._llm_thread.join(timeout=2.0)

        # 进入 AWAKE
        with self._state_lock:
            self._state = _State.AWAKE
        self._last_llm_activity = now

        print(f"[VoiceChat] 唤醒成功, 距上次 {time.time() - self._last_llm_activity:.1f}s")

        # 通知外部
        if self.on_wake_word:
            try:
                self.on_wake_word()
            except Exception as e:
                print(f"[VoiceChat] on_wake_word 异常: {e}")

    def _on_sentence(self, pcm_data: bytes):
        """VAD 断句回调：仅 AWAKE 状态时派发 LLM。"""
        duration_ms = len(pcm_data) // 2 * 1000 // self.SAMPLE_RATE
        with self._state_lock:
            if self._state != _State.AWAKE:
                return  # IDLE、LLM_PENDING 或 SPEAKING 时忽略
            if duration_ms < 200:
                short_noise = True
            else:
                short_noise = False
                # Claim the turn before leaving the audio callback so a second VAD
                # sentence cannot race with this one. Capture stays muted until the
                # corresponding TTS turn has physically finished playing.
                self._state = _State.LLM_PENDING

        callback = getattr(self, "on_speech_end", None)
        if callback:
            callback()
        if short_noise:
            return

        self._pipe.pause()

        # 转 WAV → base64，在新线程发 LLM
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="vc_")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm_data)

            debug_store = getattr(self, "_voice_debug", None)
            if debug_store is not None:
                debug_path = debug_store.save_file("llm_audio_input", wav_path)
                if debug_path is not None:
                    print(f"[VoiceChat] 已保存 LLM 音频输入: {debug_path}")

            with open(wav_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            self._dispatch_llm(audio_b64)

        except Exception as e:
            print(f"[VoiceChat] 语音编码失败: {e}")
            # Complete the empty turn so the playback node can acknowledge it
            # and reopen capture instead of leaving the microphone muted.
            self._llm_done()
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _on_speech_start(self, _initial_pcm: bytes):
        with self._state_lock:
            if self._state != _State.AWAKE:
                return
        callback = getattr(self, "on_speech_start", None)
        if callback:
            callback()

    def _on_speech_cancel(self):
        with self._state_lock:
            if self._state != _State.AWAKE:
                return
        callback = getattr(self, "on_speech_end", None)
        if callback:
            callback()

    def _on_timeout(self):
        """LLM 超时，回到 IDLE。"""
        print("[VoiceChat] LLM 40s 无回复，超时回到待机")
        self._cancel_llm.set()
        with self._state_lock:
            self._state = _State.IDLE
        self._pipe.set_awake(False)
        self._pipe.resume()
        if self.on_llm_timeout:
            try:
                self.on_llm_timeout()
            except Exception as e:
                print(f"[VoiceChat] on_llm_timeout 异常: {e}")

    def _timeout_watch(self):
        """后台线程：定期检查 LLM 超时。"""
        while True:
            time.sleep(2)
            with self._state_lock:
                state = self._state
            if state in (_State.AWAKE, _State.LLM_PENDING):
                if time.time() - self._last_llm_activity > self.LLM_IDLE_TIMEOUT:
                    self._on_timeout()

    # ================================================================
    # LLM 调度
    # ================================================================
    def _dispatch_llm(self, audio_b64: str):
        """将已编码音频派发给后台 LLM 线程。"""
        self._cancel_llm.clear()
        self._llm_thread = threading.Thread(
            target=self._send_to_llm, args=(audio_b64,), daemon=True
        )
        self._llm_thread.start()

    def _send_to_llm(self, audio_b64: str):
        """后台线程：拼 messages → 调 LLM → 流式回调。"""
        self.last_action_results = []
        audio_message = self.multimodal.build_audio_message(audio_b64)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._validated_history())
        messages.append(audio_message)

        print(f"[VoiceChat] 发送音频 → {self.model}")
        t0 = time.time()

        try:
            streamed = self._stream_tool_calls(
                messages,
                tools=get_multimodal_tools(),
                tool_choice="auto",
            )
            if streamed is None:
                return
            tool_calls, raw_content = streamed
            heard_text, response_text, expression, intensity = self._dialog_answer(tool_calls)
            intent_type = self._dialog_intent_type(tool_calls)
            structured_ok = bool(response_text)

            if not response_text:
                print(
                    "[VoiceChat] 模型漏掉 direct_answer，使用无历史、无动作工具重试"
                )
                retry_messages = [
                    {"role": "system", "content": self.system_prompt},
                    audio_message,
                ]
                retry = self._stream_tool_calls(
                    retry_messages,
                    tools=[MULTIMODAL_DIRECT_ANSWER_TOOL],
                    tool_choice={
                        "type": "function",
                        "function": {"name": DIRECT_ANSWER_TOOL_NAME},
                    },
                )
                if retry is None:
                    return
                retry_calls, retry_raw_content = retry
                heard_text, response_text, expression, intensity = self._dialog_answer(retry_calls)
                intent_type = self._dialog_intent_type(retry_calls)
                structured_ok = bool(response_text)
                if not response_text:
                    print(
                        "[VoiceChat] 结构化回答重试失败 "
                        f"(raw_content={len(raw_content) + len(retry_raw_content)} chars)"
                    )
                    response_text = self.FALLBACK_REPLY

            # Visual task type is a semantic model decision.  A second bounded
            # text-only planning pass disambiguates plain inspection from a
            # compound observe-condition-action request without parsing Chinese
            # connectors in application code.
            handled_visual_tools = set()
            proposed_actions = [
                call for call in tool_calls
                if isinstance(call, dict)
                and call.get("name") != DIRECT_ANSWER_TOOL_NAME
            ]
            visual_tool_names = {
                "inspect_camera", CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME
            }
            has_visual_proposal = any(
                call.get("name") in visual_tool_names
                for call in proposed_actions
            )
            needs_visual_replan = bool(
                structured_ok
                and heard_text
                and (
                    has_visual_proposal
                    or (intent_type == "execute_task" and not proposed_actions)
                )
            )
            visual_plan = (
                self._plan_visual_task(heard_text)
                if needs_visual_replan else None
            )
            conditional_intent = bool(
                visual_plan
                and visual_plan.get("name") == CONDITIONAL_TASK_TOOL_NAME
            )
            compound_visual_intent = bool(
                visual_plan
                and visual_plan.get("name")
                in {CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME}
            )
            if needs_visual_replan:
                non_visual_calls = [
                    call for call in tool_calls
                    if isinstance(call, dict)
                    and call.get("name") not in visual_tool_names
                ]
                if visual_plan is not None:
                    tool_calls = (
                        [visual_plan]
                        if compound_visual_intent
                        else [*non_visual_calls, visual_plan]
                    )
                    # The specialized planner resolves contradictory generic
                    # intent labels from the audio turn.
                    intent_type = "execute_task"
                else:
                    tool_calls = [
                        call for call in non_visual_calls
                        if call.get("name") == DIRECT_ANSWER_TOOL_NAME
                    ]
                    response_text = (
                        "这个任务没有形成可执行计划，所以我没有执行。"
                    )
            photo_handler = getattr(self, "on_photo_request", None)
            inspection_handler = getattr(self, "on_inspection_request", None)
            if (
                structured_ok
                and heard_text
                and visual_plan is not None
                and visual_plan.get("name") == "inspect_camera"
                and visual_plan.get("arguments", {}).get("save_photo") is True
                and not conditional_intent
                and photo_handler
            ):
                handled_visual_tools.add("inspect_camera")
                try:
                    handled_response = photo_handler()
                    if not isinstance(handled_response, str) or not handled_response.strip():
                        raise RuntimeError("拍照处理器没有返回结果")
                    response_text = handled_response.strip()
                except Exception as exc:
                    print(f"[VoiceChat] 拍照处理失败: {exc}")
                    response_text = "这次没拍成功，请检查摄像头后再试。"
            elif (
                structured_ok
                and heard_text
                and visual_plan is not None
                and visual_plan.get("name") == "inspect_camera"
                and visual_plan.get("arguments", {}).get("save_photo") is not True
                and inspection_handler
            ):
                handled_visual_tools.add("inspect_camera")
                try:
                    handled_response = inspection_handler(heard_text)
                    if not isinstance(handled_response, str) or not handled_response.strip():
                        raise RuntimeError("视觉查看处理器没有返回结果")
                    response_text = handled_response.strip()
                except Exception as exc:
                    print(f"[VoiceChat] 视觉查看失败: {exc}")
                    response_text = "这次没看清，请检查摄像头后再试。"

            pending_actions = []
            for tc in tool_calls:
                if self._cancel_llm.is_set():
                    return
                if not structured_ok:
                    break
                if tc["name"] == DIRECT_ANSWER_TOOL_NAME:
                    continue
                if intent_type in {"conversation", "capability_query"}:
                    print(
                        "[VoiceChat] 非执行意图忽略动作工具: "
                        f"intent={intent_type}, tool={tc['name']}"
                    )
                    continue
                if compound_visual_intent and tc["name"] not in {
                    CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME
                }:
                    print(
                        "[VoiceChat] 忽略被拆分的复合任务工具: "
                        f"{tc['name']}"
                    )
                    continue
                if tc["name"] in handled_visual_tools:
                    continue
                arguments = dict(tc["arguments"])
                # Older providers may still return this retired diagnostic
                # field.  It is not used to infer or authorize intent.
                arguments.pop("grounding", None)
                pending_actions.append({
                    "name": tc["name"],
                    "arguments": arguments,
                })

            if pending_actions and structured_ok:
                action_state = self._execute_action_sequence(
                    heard_text,
                    pending_actions,
                )
                self.last_action_results = action_state.get("results", [])
                for result in self.last_action_results:
                    print(
                        "[VoiceChat] 工具结果: "
                        f"{result.get('name') or result.get('action') or '<plan>'} "
                        f"-> {result.get('status')}"
                        + (
                            f" ({result.get('reason')})"
                            if result.get("reason") else ""
                        )
                    )
                    if result.get("status") == "completed":
                        handled_response = result.get("response")
                        # A semantic skill such as inspect_camera may perform a
                        # second model request and replace the acknowledgement.
                        if isinstance(handled_response, str) and handled_response.strip():
                            response_text = handled_response.strip()
                        continue
                    if (
                        result.get("status") == "rejected"
                        and result.get("reason") == "non_command_context"
                    ):
                        # The guard correctly blocked a capability question or
                        # other non-command. Keep the model's conversational
                        # answer; no robot action was attempted.
                        continue
                    if result.get("status") != "skipped":
                        response_text = "这个动作没有确认完成，我已停止后续动作。"
                        break

            if self._cancel_llm.is_set():
                return
            expression_callback = getattr(self, "on_expression", None)
            if expression_callback:
                expression_callback(expression, intensity)
            if self.on_llm_chunk:
                self.on_llm_chunk(response_text)

            elapsed = time.time() - t0
            reply = response_text
            print(f"[VoiceChat] LLM 回复 ({elapsed:.1f}s): {reply}")

            if heard_text and response_text != self.FALLBACK_REPLY:
                self._append_history_turn(heard_text, response_text)
                print(f"[VoiceChat] 听写: {heard_text}")

            # 通知外部完整回复
            if self.on_llm_reply:
                self.on_llm_reply(reply)

        except Exception as e:
            print(f"[VoiceChat] LLM 调用失败: {e}")
        finally:
            self._llm_done()

    def _execute_action_sequence(self, heard_text, actions):
        """Prefer the native tree callback, then use the compatibility tree."""
        native_handler = getattr(self, "on_action_plan", None)
        if native_handler is not None:
            native_state = native_handler(heard_text or "", actions)
            if native_state is not None:
                return native_state

        workflow = getattr(self, "_behavior_tree_workflow", None)
        if workflow is None:
            workflow = BehaviorTreeActionWorkflow(
                authorize=lambda _prompt, name, arguments: (
                    validate_action_arguments(name, arguments)
                ),
                execute=self._execute_tool_call,
                cancelled=self._cancel_llm.is_set,
            )
            self._behavior_tree_workflow = workflow
        return workflow.invoke(
            turn_id="multimodal_dialog",
            user_prompt=heard_text or "",
            actions=actions,
        )

    def _execute_tool_call(self, name, arguments):
        callback = getattr(self, "on_tool_call", None)
        if callback is None:
            return {
                "status": "failed",
                "action": name,
                "reason": "action_handler_unavailable",
            }
        self._last_llm_activity = time.time()
        try:
            return callback(name, arguments)
        finally:
            self._last_llm_activity = time.time()

    def _plan_visual_task(self, heard_text):
        """Semantically select inspection, conditional action, or search."""
        plan_tools = [
            tool for tool in get_action_tools()
            if tool.get("function", {}).get("name")
            in {"inspect_camera", CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME}
        ]
        if len(plan_tools) != 3 or self._cancel_llm.is_set():
            return None
        messages = [
            {"role": "system", "content": (
                "你是机器人视觉任务规划器。直接理解用户完整语义，不依赖固定连接词。"
                "如果用户只要求查看当前画面，调用 inspect_camera；如果用户要求先观察"
                "画面、再根据观察结果决定是否执行动作，调用 run_conditional_task。"
                "用户要求在房间、周围等大于单个视野的环境中寻找或定位目标时，应把它"
                "理解为主动视觉搜索并调用 search_environment；搜索循环和转向由行为树"
                "负责，不要把搜索改写成否定条件任务。若用户还要求在目标找到后继续"
                "执行一个或多个动作，把这些动作按顺序作为 search_environment 的"
                "on_found_actions 返回；不要在应用层按中文连接词拆分，也不要在目标"
                "尚未确认找到前单独调用这些动作工具。"
                "如果不是视觉任务，不调用工具。只能选择一种任务，不观察环境、不执行"
                "动作，也不声称已经完成。"
            )},
            {"role": "user", "content": heard_text},
        ]
        def select_valid_plan(calls):
            invalid_search = False
            for call in calls:
                name = call.get("name")
                if name not in {
                    "inspect_camera", CONDITIONAL_TASK_TOOL_NAME, VISUAL_SEARCH_TOOL_NAME
                }:
                    continue
                raw_arguments = call.get("arguments")
                if not isinstance(raw_arguments, dict):
                    continue
                arguments = dict(raw_arguments)
                arguments.pop("grounding", None)
                allowed, reason = validate_action_arguments(name, arguments)
                if allowed:
                    return {"name": name, "arguments": arguments}, False
                if name == VISUAL_SEARCH_TOOL_NAME:
                    invalid_search = True
                print(f"[VoiceChat] 视觉计划无效: {name} ({reason})")
            return None, invalid_search

        try:
            streamed = self._stream_tool_calls(messages, tools=plan_tools, tool_choice="auto")
            if streamed is None:
                return None
            calls, _content = streamed
            plan, invalid_search = select_valid_plan(calls)
            if plan is not None:
                return plan
            if invalid_search and not self._cancel_llm.is_set():
                search_tool = next(
                    tool for tool in plan_tools
                    if tool.get("function", {}).get("name") == VISUAL_SEARCH_TOOL_NAME
                )
                repair_messages = [
                    {"role": "system", "content": (
                        "修复上一轮视觉搜索工具参数。只能调用 search_environment。"
                        "on_found_actions 必须是数组；每个元素只能包含 name 和 arguments，"
                        "并且只有在找到目标后才执行。不得输出普通文本或额外字段。"
                    )},
                    {"role": "user", "content": heard_text},
                ]
                repaired = self._stream_tool_calls(
                    repair_messages,
                    tools=[search_tool],
                    tool_choice={
                        "type": "function",
                        "function": {"name": VISUAL_SEARCH_TOOL_NAME},
                    },
                )
                if repaired is not None:
                    repaired_plan, _invalid = select_valid_plan(repaired[0])
                    if repaired_plan is not None:
                        return repaired_plan
        except Exception as exc:
            print(f"[VoiceChat] 视觉任务规划失败: {exc}")
        return None

    def analyze_image(self, question: str, image_base64: str) -> str:
        """Analyze one camera JPEG using the configured multimodal model.

        This request exposes only the trusted direct_answer outlet.  It is
        called after the audio model semantically selects inspect_camera, so
        camera activation remains voice-driven.
        """
        prompt = (
            "请根据附带的摄像头画面回答问题。只依据图片内容；看不清时明确说看不清。"
            "回答必须简短、自然、适合直接播报。\n"
            f"用户问题：{(question or '看看当前画面').strip()}"
        )
        messages = [
            {
                "role": "system",
                "content": with_structured_answer_policy(
                    with_direct_speech_policy(
                        "你是瓦力的视觉，只负责观察当前摄像头图片并回答问题。"
                    )
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]
        streamed = self._stream_tool_calls(
            messages,
            tools=[DIRECT_ANSWER_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": DIRECT_ANSWER_TOOL_NAME},
            },
        )
        if streamed is None:
            raise RuntimeError("视觉分析请求被中断")
        tool_calls, _raw_content = streamed
        _heard_text, response_text, _expression, _intensity = self._dialog_answer(tool_calls)
        if not response_text:
            raise RuntimeError("视觉模型没有返回 direct_answer.response")
        return response_text

    def evaluate_image_condition(
        self,
        observation: str,
        condition: str,
        image_base64: str,
    ) -> dict[str, str]:
        """Return a closed structured decision for a conditional task image."""
        prompt = (
            "只依据附带的当前摄像头画面，判断给出的条件本身是否成立。必须调用 "
            "conditional_decision；decision 只能是 yes、no、uncertain。条件若描述"
            "‘当前画面未看到/不可见某目标’，在画面清晰且确实没有可识别目标时应返回 yes；"
            "不要把‘当前画面不可见’误解成‘目标在整个房间不存在’。只有画质、遮挡或目标"
            "歧义导致连当前画面可见性也无法判断时才返回 uncertain。"
            "不要执行动作。\n"
            f"观察任务：{observation}\n判断条件：{condition}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是机器人视觉条件判断器。只能依据当前图片调用 "
                    "conditional_decision。严格判断条件的字面范围，区分‘当前画面不可见’"
                    "与‘现实环境中不存在’；前者可以从清晰的当前画面直接判断。"
                    "真正无法判断时返回 uncertain，禁止猜测。"
                    "不得输出台词、JSON 文本、思考过程或执行任何动作。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]
        decision_tool = conditional_decision_tool()
        streamed = self._stream_tool_calls(
            messages,
            tools=[decision_tool],
            tool_choice={
                "type": "function",
                "function": {"name": CONDITIONAL_DECISION_TOOL_NAME},
            },
        )
        if streamed is None:
            raise RuntimeError("视觉条件判断请求被中断")
        tool_calls, _raw_content = streamed
        for call in tool_calls:
            if call.get("name") != CONDITIONAL_DECISION_TOOL_NAME:
                continue
            decision = parse_conditional_decision(call.get("arguments"))
            if decision.get("evidence") in {
                "invalid_model_output", "invalid_model_decision"
            }:
                break
            return decision
        raise RuntimeError("视觉模型没有返回有效的 conditional_decision")

    def evaluate_visual_search(
        self,
        target: str,
        question: str,
        image_base64: str,
    ) -> dict[str, str]:
        """Return a closed found/not-found result for one search-tree view."""
        prompt = (
            "只检查附带的当前摄像头图片，不考虑画面外或遮挡后的可能性。"
            "如果当前图片清楚显示搜索目标，status=found，并在 response 中说明位置；"
            "当前图片清晰但没有目标时 status=not_found；只有画面本身无法判断时才用"
            " uncertain。必须调用 visual_search_result，不执行动作。\n"
            f"搜索目标：{target}\n用户问题：{question}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是行为树的视觉检测叶节点。只判断目标在当前图片中是否可见，"
                    "必须返回结构化 visual_search_result，不规划动作，不输出普通文本。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]
        tool = visual_search_result_tool()
        streamed = self._stream_tool_calls(
            messages,
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": VISUAL_SEARCH_RESULT_TOOL_NAME},
            },
        )
        if streamed is None:
            raise RuntimeError("视觉搜索请求被中断")
        calls, _raw_content = streamed
        for call in calls:
            if call.get("name") != VISUAL_SEARCH_RESULT_TOOL_NAME:
                continue
            result = parse_visual_search_result(call.get("arguments"))
            if result is not None:
                return result
        raise RuntimeError("视觉模型没有返回有效的 visual_search_result")

    def _stream_tool_calls(self, messages, *, tools, tool_choice):
        """Return parsed tool calls and untrusted content for one LLM request."""
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "modalities": ["text"],
            "tools": tools,
            "tool_choice": normalize_tool_choice(self.llm_settings, tool_choice),
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": self.API_TIMEOUT,
            "max_tokens": self.max_tokens,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3,
        }
        request_kwargs.update(reasoning_request_options(self.llm_settings))
        response = self.client.chat.completions.create(**request_kwargs)
        accumulator = ToolCallAccumulator()
        raw_content = []
        for chunk in response:
            if self._cancel_llm.is_set():
                print("[VoiceChat] LLM 调用被中断")
                if hasattr(response, "close"):
                    try:
                        response.close()
                    except Exception:
                        pass
                return None
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            accumulator.feed(delta)
            if delta.content:
                raw_content.append(delta.content)
            self._last_llm_activity = time.time()
        return accumulator.flush(), "".join(raw_content).strip()

    @staticmethod
    def _direct_answer(tool_calls):
        heard_text, response_text, _expression, _intensity = (
            VoiceChatService._dialog_answer(tool_calls)
        )
        return heard_text, response_text

    @staticmethod
    def _dialog_intent_type(tool_calls):
        for call in reversed(tool_calls):
            if call.get("name") != DIRECT_ANSWER_TOOL_NAME:
                continue
            value = call.get("arguments", {}).get("intent_type")
            if value in {"conversation", "capability_query", "execute_task"}:
                return value
        return ""

    @staticmethod
    def _dialog_answer(tool_calls):
        heard_text = ""
        response_text = ""
        expression, intensity = "neutral", "low"
        for call in tool_calls:
            if call["name"] != DIRECT_ANSWER_TOOL_NAME:
                continue
            arguments = call["arguments"]
            heard = arguments.get("heard_text")
            response = arguments.get("response")
            if isinstance(heard, str):
                heard_text = heard.strip()[:240]
            if isinstance(response, str):
                response_text = response.strip()
            expression, intensity = normalize_expression(
                arguments.get("expression"), arguments.get("intensity")
            )
        return heard_text, response_text, expression, intensity

    def _validated_history(self):
        history = list(self._chat_history)
        cleaned = []
        expected = "user"
        for message in history:
            if not isinstance(message, dict) or message.get("role") != expected:
                print("[VoiceChat] 检测到不成对的旧对话历史，已清空")
                self._chat_history.clear()
                return []
            content = message.get("content")
            if isinstance(content, list):
                # Keep only transcript text.  Images/audio are current-turn
                # inputs and must never survive into conversational context.
                content = "\n".join(
                    block["text"].strip()
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].strip()
                )
            if not isinstance(content, str):
                print("[VoiceChat] 检测到非文本旧对话历史，已清空")
                self._chat_history.clear()
                return []
            cleaned.append({"role": message["role"], "content": content})
            expected = "assistant" if expected == "user" else "user"
        if expected != "user":
            print("[VoiceChat] 检测到未完成的旧对话回合，已清空")
            self._chat_history.clear()
            return []
        if cleaned != history:
            self._chat_history.clear()
            self._chat_history.extend(cleaned)
        return cleaned

    def _append_history_turn(self, heard_text, response_text):
        self._validated_history()
        self._chat_history.append({"role": "user", "content": heard_text})
        self._chat_history.append({"role": "assistant", "content": response_text})

    def _llm_done(self):
        """LLM 调用结束，等待扬声器真正播完后再恢复采集。"""
        self._last_llm_activity = time.time()
        with self._state_lock:
            if self._state == _State.LLM_PENDING:
                self._state = _State.SPEAKING
        if self.on_llm_done:
            try:
                self.on_llm_done()
            except Exception as e:
                print(f"[VoiceChat] on_llm_done 异常: {e}")
