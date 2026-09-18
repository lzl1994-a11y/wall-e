#!/usr/bin/env python3
"""LLM Brain ROS Node / 大语言模型大脑 ROS 节点.

English:
    Central dialogue, vision, and action orchestration node for the Wall-E robot.
    Subscribes to recognized voice text (ASR), delegates natural language understanding
    and streaming dialogue generation to LLMService, coordinates streaming sub-clause
    TTS speech output and screen displays, dispatches physical action commands
    (via direct action publisher or Behavior Tree execution), and manages camera
    interaction workflows (visual QA, conditional tasks, and photo shooting).

中文:
    Wall-E 机器人的核心对话、视觉感知与动作编排 ROS 节点。
    订阅语音识别文本 (ASR)，将自然语言理解与流式对话生成委托给 LLMService；
    协调流式从句级 TTS 语音合成输出与屏幕文本展示；
    分发底层实体动作指令（支持直接动作发布与行为树调度执行）；
    驱动相机视觉交互工作流（包括视觉问答、视觉条件任务与拍照存盘）。
"""

import json
import queue
import re
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8MultiArray

# Action, plan, and Behavior Tree protocol & execution services.
# 动作、计划以及行为树协议与执行服务。
from services.action_acknowledgement import action_acknowledgement
from services.action_command import ACTION_REQUEST_TOPIC
from services.action_execution import CorrelatedActionExecutor
from services.behavior_tree_execution import CorrelatedPlanExecutor
from services.native_plan_execution import NativePlanExecutionAdapter
from services.ros2_action_execution import (
    Ros2ActionPlanExecutor,
    create_wali_task_action_client,
)
from services.behavior_tree_protocol import (
    BEHAVIOR_TREE_CANCEL_TOPIC,
    BEHAVIOR_TREE_EXECUTE_TOPIC,
    BEHAVIOR_TREE_STATUS_TOPIC,
)
from services.action_intent_guard import (
    validate_action_arguments,
    validate_action_call,
)
from services.action_status import ACTION_STATUS_TOPIC

# LLM core, policies, streaming, and conversation history services.
# 大模型核心调用、策略清洗、流式响应与对话历史服务。
from services.llm_service import LLMService
from services.llm_response_policy import LLMResponsePolicy
from services.llm_stream_response import StreamResponseAccumulator
from services.llm_voice_turn import LLMVoiceTurnState
from services.llm_conversation_history import LLMConversationHistory
from services.llm_request_preparation import (
    ROUTE_CAMERA_INSPECTION,
    ROUTE_CAMERA_PHOTO,
    ROUTE_CONDITIONAL_TASK,
    ROUTE_SAFETY_ACTION,
    prepare_voice_request,
)
from services.llm_tool_proposal import evaluate_tool_proposal
from services.llm_conditional_planning import (
    build_conditional_fallback_request,
    evaluate_conditional_plan,
    evaluate_native_conditional_event,
    parse_conditional_fallback_json,
)
from services.llm_visual_request import (
    VisualResponseAccumulator,
    build_camera_qa_request,
    build_conditional_vision_request,
    build_game_vision_request,
)
from services.llm_empty_answer_retry import (
    EmptyAnswerRetryAccumulator,
    build_empty_answer_retry_request,
)
from services.camera_frame import save_camera_photo
from services.llm_photo_capture import (
    decide_photo_preview,
    photo_save_failed,
    photo_save_succeeded,
)

# Game protocol and streaming services.
# 游戏协议与视频流服务。
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_STATE_TOPIC,
    decode_game_frame,
    game_mode_from_message,
)
from services.game_tft_stream import prepare_game_bgr
from services.game_commentary import GameCommentaryController

# TFT preview, audio, expression, and workflow services.
# TFT 屏幕预览、语音协议、微表情与业务工作流服务。
from services.tft_preview_client import TftPreviewClient
from services.tft_preview_server import load_tft_preview_settings
from services.tts_protocol import encode_turn_end
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TOPIC,
    encode_dialog_expression,
)
from services.conditional_task import (
    CONDITIONAL_TASK_TOOL_NAME,
    build_conditional_task_failure_outcome,
    build_conditional_task_outcome,
)
from services.llm_action_plan import LLMActionPlanWorkflow
from services.dialog_workflow import CameraInspectionWorkflow, ConditionalTaskWorkflow
from services.voice_debug import RollingVoiceDebugStore


class LLMBrainNode(Node):
    """ROS 2 Node for LLM dialogue, vision, and action orchestration.

    ROS 2 节点：负责大语言模型多模态对话、视觉感知与动作编排。
    """

    # Maximum dialogue messages retained in the rolling prompt history.
    # 发送给大模型的滑动对话历史最大条数。
    CHAT_HISTORY_MESSAGES = 12

    # Regex to detect long-form generation requests (e.g. storytelling).
    # 用于识别长文生成/讲故事类请求的正则表达式。
    LONG_FORM_REQUEST_RE = LLMResponsePolicy.LONG_FORM_REQUEST_RE

    # Token budget override for long-form requests.
    # 长文生成请求的最大 token 预算上限。
    LONG_FORM_MAX_TOKENS = LLMResponsePolicy.LONG_FORM_MAX_TOKENS

    # Minimum character length before emitting the first sub-clause TTS sentence.
    # 首句流式 TTS 播报的最小字符长度阈值（避免过早短音播报断断续续）。
    FIRST_TTS_CLAUSE_MIN_CHARS = 10

    # Punctuations that trigger a sub-clause split during streaming TTS.
    # 触发流式从句切分的标点符号集合。
    CLAUSE_PUNCTUATIONS = {'，', ',', '；', ';', '：', ':'}

    # Labels indicating ASR correction prefix in model responses.
    # 模型输出中识别 ASR 纠错的前缀标签集合。
    CORRECTION_LABELS = LLMResponsePolicy.CORRECTION_LABELS

    # Regex to sanitize text for speech synthesis (strips Markdown/unsupported chars).
    # 用于清洗语音合成文本中 Markdown 标记和不支持符号的正则表达式。
    TTS_CLEAN_RE = LLMResponsePolicy.TTS_CLEAN_RE

    # Regex to match output line prefixes.
    # 匹配输出行首前缀的正则表达式。
    OUTPUT_LINE_PREFIX_RE = LLMResponsePolicy.OUTPUT_LINE_PREFIX_RE

    def __init__(self):
        """Initialize the LLMBrainNode, publishers, subscriptions, and worker thread.

        初始化 LLMBrainNode 节点、各话题发布器/订阅器及后台工作线程。
        """
        super().__init__('walle_llm_brain')

        # Core service clients and workflow holders.
        # 核心服务客户端及工作流持有者。
        self.llm = None
        self._camera_inspection_workflow = None
        self._conditional_task_workflow = None
        self._llm_action_plan_workflow = None

        # Executors for physical actions and Behavior Tree execution.
        # 实体动作执行器与行为树执行器。
        self._action_executor = CorrelatedActionExecutor()
        self._behavior_tree_executor = CorrelatedPlanExecutor()
        self._ros2_task_executor = None

        # Debug store, conversation history deque, and punctuation boundaries.
        # 语音调试存储、滑动对话历史双端队列与整句断句标点集合。
        self._voice_debug = RollingVoiceDebugStore()
        self.chat_history = deque(maxlen=24)
        self.punctuations = {'。', '？', '.', '?', '！', '!'}

        # Worker queue and thread lifecycle flags.
        # 任务请求队列与工作线程生命周期标记。
        self._request_queue = queue.Queue(maxsize=8)
        self._worker_running = False

        # Game commentary mode and controller.
        # 游戏解说模式与控制器。
        self._game_mode = "robot"
        self._game_commentary = GameCommentaryController()

        # Create ROS endpoints before the slow LLM client init. This lets DDS
        # discover `voice_text` while the model service is warming up.
        # 在耗时较长的大模型客户端初始化之前先创建 ROS 通信端点，使 DDS 能在模型预热期间完成话题发现。
        self.voice_subscription = self.create_subscription(
            String,
            'voice_text',
            self.voice_callback,
            10,
        )
        self.tts_publisher = self.create_publisher(String, 'tts_text', 10)
        self.action_publisher = self.create_publisher(String, ACTION_REQUEST_TOPIC, 10)
        self.behavior_tree_publisher = self.create_publisher(
            String, BEHAVIOR_TREE_EXECUTE_TOPIC, 10
        )
        self.behavior_tree_cancel_publisher = self.create_publisher(
            String, BEHAVIOR_TREE_CANCEL_TOPIC, 10
        )
        self.create_subscription(
            String,
            BEHAVIOR_TREE_STATUS_TOPIC,
            self._on_behavior_tree_status,
            10,
        )
        try:
            # Initialize ROS 2 BehaviorTree Action client.
            # 初始化行为树 ROS 2 Action 客户端。
            action_client, goal_type = create_wali_task_action_client(
                self, Path(__file__).resolve().parent.parent
            )
            self._ros2_task_executor = Ros2ActionPlanExecutor(action_client, goal_type)
            self.get_logger().info('BehaviorTree ROS2 Action client is ready.')
        except Exception as exc:
            self.get_logger().warning(
                f'BehaviorTree ROS2 Action client unavailable; legacy fallback remains active: {exc}'
            )

        # Native plan execution adapter bridging ROS 2 and legacy BT execution.
        # 原生计划执行适配器，统一桥接 ROS 2 Action 与旧版行为树执行通道。
        self._native_plan_adapter = NativePlanExecutionAdapter(
            ros2_executor=self._ros2_task_executor,
            legacy_executor=self._behavior_tree_executor,
            publish=lambda payload: self.behavior_tree_publisher.publish(
                String(data=payload)
            ),
            cancel_publish=lambda payload: self.behavior_tree_cancel_publisher.publish(
                String(data=payload)
            ),
            owner_available=lambda: (
                self.behavior_tree_publisher.get_subscription_count() > 0
            ),
        )

        # Action execution feedback subscriber.
        # 动作执行状态反馈话题订阅。
        self.create_subscription(
            String,
            ACTION_STATUS_TOPIC,
            self._on_action_status,
            10,
        )

        # Dialogue topic publishers.
        # 对话相关话题发布器。
        self.corrected_publisher = self.create_publisher(String, 'corrected_text', 10)
        self.full_ai_publisher = self.create_publisher(String, 'full_ai_text', 10)
        self.screen_dialog_publisher = self.create_publisher(String, 'screen_dialog', 10)
        self.busy_publisher = self.create_publisher(String, 'llm_busy', 10)
        self.dialog_expression_publisher = self.create_publisher(
            String, DIALOG_EXPRESSION_TOPIC, 10
        )

        # TFT preview client for camera streaming.
        # TFT 屏幕相机预览客户端配置与初始化。
        self.tft_preview_settings = load_tft_preview_settings()
        self.tft_preview = TftPreviewClient(self, logger=self.get_logger())

        # Game mode and frame subscriptions, plus periodic commentary timer.
        # 游戏模式与图像帧订阅，以及周期性游戏解说定时器。
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_timer(1.0, self._game_commentary_tick)

        # Initialize LLMService.
        # 初始化大模型服务。
        try:
            self.llm = LLMService()
            self.get_logger().info('LLM service initialized.')
        except Exception as e:
            self.get_logger().error(f'LLM service initialization failed: {e}')
            return

        # Start dedicated background worker thread.
        # 启动独立的后台工作线程，防止大模型 I/O 阻塞 ROS 回调。
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._llm_worker,
            name='llm-worker',
            daemon=True,
        )
        self._worker_thread.start()

    def _on_game_state(self, message):
        """Update game mode state and clean up commentary when exiting playing mode.

        更新游戏模式状态；当退出游戏运行状态时清理并完结解说。
        """
        mode = game_mode_from_message(message.data)
        if mode is None:
            return
        previous = self._game_mode
        self._game_mode = mode
        self._game_commentary.set_mode(mode)
        if mode != "robot":
            return
        if previous != "robot":
            self._game_commentary.finish_commentary()

    def _on_game_frame(self, message):
        """Buffer incoming game video frames for commentary inference.

        接收并缓存游戏视频画面帧，供解说推理调度使用。
        """
        frame = decode_game_frame(bytes(message.data))
        if frame is None:
            return
        self._game_commentary.accept_frame(frame)

    def _game_commentary_tick(self):
        """Periodic timer callback to extract a due game frame and enqueue vision task.

        定时器周期回调：提取到达解说间隔的游戏画面帧，编码后入队视觉推理任务。
        """
        frame = self._game_commentary.take_due_frame()
        if frame is None or self.llm is None:
            if frame is not None:
                self._game_commentary.finish_commentary()
            return
        import numpy as np

        raw, width, height, pitch = frame
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, pitch // 4, 4)
        jpeg = prepare_game_bgr(image[:, :width, :3], quality=75)
        if not jpeg:
            self._game_commentary.finish_commentary()
            return
        try:
            self._request_queue.put_nowait({
                'kind': 'game_vision',
                'turn_id': 'game-' + uuid.uuid4().hex[:8],
                'jpeg': jpeg,
            })
        except queue.Full:
            self._game_commentary.finish_commentary()

    def voice_callback(self, msg):
        """Queue the request so the ROS callback thread is never blocked by LLM I/O.

        语音识别输入回调：过滤噪声音译，即时更新屏幕UI，并将请求入队，避免阻塞 ROS 回调线程。
        """
        if self._game_mode != "robot":
            self.get_logger().info("游戏模式热备中，忽略语音 LLM 输入")
            return
        user_prompt = (msg.data or '').strip()
        # Filter common ASR noise/transcription artifacts (e.g. '#' or standalone punctuation).
        # 过滤掉常见的 ASR 噪声音译（如 #，或者单纯的标点符号）。
        if not user_prompt or user_prompt == '#' or len(user_prompt.strip('.,?!。，？！# ')) == 0:
            self.get_logger().info(f'Ignored empty/noise ASR input: "{user_prompt}"')
            return

        turn_id = uuid.uuid4().hex[:12]
        self.get_logger().info(f'[{turn_id}] Voice text received: {user_prompt}')

        # Show what was heard immediately. Waiting for LLM/tool completion made
        # the user's text appear frozen during a slow model response.
        # 立即在屏幕上展示识别到的文字，避免模型慢响应时用户界面出现停滞感。
        self._publish_screen_dialog(turn_id, user_prompt, '', [])

        try:
            self._request_queue.put_nowait({
                'turn_id': turn_id,
                'user_prompt': user_prompt,
            })
        except queue.Full:
            self.get_logger().error('LLM request queue is full; dropped this voice input.')

    def _llm_worker(self):
        """Background worker thread consuming and executing dialogue and vision tasks.

        后台工作线程：循环消费并执行请求队列中的语音对话与游戏视觉任务。
        """
        while self._worker_running:
            try:
                task = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                self._request_queue.task_done()
                continue

            try:
                if task.get('kind') == 'game_vision':
                    self._process_game_vision_task(task)
                else:
                    self._process_voice_task(task['turn_id'], task['user_prompt'])
            except Exception as e:
                self.get_logger().error(f'Unhandled LLM worker error: {e}\n{traceback.format_exc()}')
                failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
                self._publish_tts(failure_text, task.get('turn_id', ''))
                self._publish_screen_dialog(
                    task.get('turn_id', ''),
                    task.get('user_prompt', ''),
                    failure_text,
                    [],
                    error=str(e),
                )
                self._finish_tts_turn(task.get('turn_id', ''))
            finally:
                if task.get('kind') == 'game_vision':
                    self._game_commentary.finish_commentary()
                self._request_queue.task_done()

    def _process_game_vision_task(self, task):
        """Process a game vision commentary task via multimodal LLM streaming.

        处理游戏视觉解说任务：调用多模态模型流式理解画面并播报解说。
        """
        if self._game_mode != "playing":
            return
        turn_id = task['turn_id']
        self.busy_publisher.publish(String(data="busy"))
        try:
            request = build_game_vision_request(task['jpeg'])
            accumulator = VisualResponseAccumulator()
            for data in self.llm.chat_stream(
                request.prompt,
                request.history,
                image_base64=request.image_base64,
                tools_enabled=request.tools_enabled,
                structured_answer=request.structured_answer,
                system_prompt=request.system_prompt,
                max_tokens_override=request.max_tokens_override,
            ):
                accumulator.process_event(data)
            answer = accumulator.clean_answer()
            if not answer:
                raise RuntimeError('游戏视觉模型返回空答案')
            self._publish_tts(answer, turn_id)
            self.full_ai_publisher.publish(String(data=answer))
        except Exception as exc:
            self.get_logger().error(f'[{turn_id}] Game vision failed: {exc}')
        finally:
            self._finish_tts_turn(turn_id)

    def _process_voice_task(self, turn_id, user_prompt):
        """Orchestrate a complete voice dialogue turn from routing to execution.

        编排单个语音对话回合的完整生命周期：
        1. 发送 busy 抑制 ASR 拾音；
        2. 意图分流（安全急停、条件任务、拍照、视觉问答、普通对话）；
        3. 流式调用 LLM 并逐从句播报 TTS 与微表情；
        4. 解析校验工具调用（Function Calling）；
        5. 落地实体动作序列；
        6. 空回复单次重试、对话历史写回与屏幕显示更新。
        """
        # Notify STT node to pause speech recognition while processing.
        # 通知 STT 节点暂停 ASR 拾音，避免机器人自言自语被误拾取。
        busy_msg = String()
        busy_msg.data = "busy"
        self.busy_publisher.publish(busy_msg)

        model_settings = getattr(self.llm, 'settings', {})
        prepared_request = prepare_voice_request(
            user_prompt,
            model_settings=model_settings,
        )

        # Route 1: Deterministic safety action (immediate stop without LLM inference).
        # 分流 1：确定性安全动作（急停直通执行，无需经过大模型推理）。
        if prepared_request.route == ROUTE_SAFETY_ACTION:
            self._process_deterministic_safety_action(
                turn_id,
                user_prompt,
                prepared_request.safety_action_name,
                prepared_request.safety_action_arguments,
            )
            return

        # Route 2: Compound conditional task (observe-condition-act workflow).
        # 分流 2：复合条件任务规划（观察-判断-执行复合工作流）。
        if prepared_request.route == ROUTE_CONDITIONAL_TASK:
            self._process_conditional_task_request(turn_id, user_prompt)
            return

        # Route 3: Camera photo capture (confirm -> TFT preview -> save to disk).
        # 分流 3：相机拍照存盘（语音确认 -> 屏幕预览 -> 保存图片）。
        if prepared_request.route == ROUTE_CAMERA_PHOTO:
            self._process_camera_photo(turn_id, user_prompt)
            return

        # Route 4: Camera visual inspection (preview frame -> visual QA).
        # 分流 4：视觉观察检测（捕获画面 -> 视觉模型问答）。
        if prepared_request.route == ROUTE_CAMERA_INSPECTION:
            self._process_camera_inspection(turn_id, user_prompt)
            return

        # Route 5: Ordinary conversational turn with optional control tools.
        # 分流 5：普通语音对话回合（可选挂载控制工具）。
        augmented_prompt = prepared_request.augmented_prompt
        tools_enabled = prepared_request.tools_enabled
        max_tokens_override = prepared_request.max_tokens_override

        self.get_logger().info(f'[{turn_id}] Sending request to LLM...')
        self.get_logger().info(f'[{turn_id}] Control tools enabled for semantic handling.')
        if max_tokens_override is not None:
            self.get_logger().info(
                f'[{turn_id}] Long-form request detected; max_tokens={max_tokens_override}.'
            )

        # Initialize turn streaming accumulator and state machine.
        # 初始化对话回合流式累加器与状态机。
        state = LLMVoiceTurnState(
            user_prompt,
            punctuations=getattr(self, 'punctuations', {'。', '？', '.', '?', '！', '!'}),
            clause_punctuations=getattr(self, 'CLAUSE_PUNCTUATIONS', {'，', ',', '；', ';', '：', ':'}),
            first_tts_clause_min_chars=getattr(self, 'FIRST_TTS_CLAUSE_MIN_CHARS', 10),
        )

        def publish_corrected(value):
            # Publish normalized ASR text on the corrected_text topic.
            # 发布归一化纠错后的文本到 corrected_text 话题。
            corrected = state.normalize_corrected_text(value)
            msg = String()
            msg.data = corrected
            self.corrected_publisher.publish(msg)
            self.get_logger().info(
                f'[{turn_id}] Corrected text: raw="{user_prompt}" corrected="{corrected}"'
            )

        def publish_spoken(value):
            # Publish streaming TTS clause and ensure a default expression is shown.
            # 发布流式 TTS 从句播报，并确保首个播报句前触发默认表情。
            if state.needs_default_expression:
                expression_publisher = getattr(
                    self, "dialog_expression_publisher", None
                )
                if expression_publisher is not None:
                    expression_publisher.publish(String(
                        data=encode_dialog_expression("neutral", "low", turn_id)
                    ))
                state.mark_expression_published()
            spoken = self._publish_tts(value, turn_id)
            if spoken:
                state.record_spoken(spoken)

        # Correction metadata is an internal concern. Publish the ASR text for
        # the existing topic contract and ask the model for speech only.
        # 纠错元数据为内部机制。对外发布 ASR 文本，大模型仅生成播报正文。
        publish_corrected(user_prompt)

        history = self._history_for_request()
        debug_store = getattr(self, "_voice_debug", None)
        if debug_store is not None:
            debug_path = debug_store.save_json("llm_input", {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "turn_id": turn_id,
                "raw_asr_text": user_prompt,
                "prompt": augmented_prompt,
                "history": history,
                "tools_enabled": tools_enabled,
                "max_tokens_override": max_tokens_override,
            })
            if debug_path is not None:
                self.get_logger().info(f'[{turn_id}] Saved LLM input: {debug_path}')

        try:
            stream = self.llm.chat_stream(
                augmented_prompt,
                history,
                tools_enabled=tools_enabled,
                max_tokens_override=max_tokens_override,
            )

            for data in stream:
                decision = state.process_event(data)

                # Streaming TTS sub-clause output.
                # 流式 TTS 从句播报。
                for tts_text in decision.tts_sentences:
                    publish_spoken(tts_text)

                # Facial expression synchronized with dialog.
                # 对话微表情同步发布。
                if decision.expression:
                    self.dialog_expression_publisher.publish(String(
                        data=encode_dialog_expression(
                            decision.expression.get('expression'),
                            decision.expression.get('intensity'),
                            turn_id,
                        )
                    ))
                    state.mark_expression_published()

                # Tool proposal validation and routing.
                # 工具调用提议校验与路由。
                if decision.tool_call:
                    proposal = evaluate_tool_proposal(
                        decision.tool_call,
                        user_prompt=user_prompt,
                        conditional_request=prepared_request.is_conditional_task,
                    )
                    if proposal.is_rejected:
                        state.record_rejected_action(
                            proposal.action_name, proposal.rejection_reason
                        )
                        if proposal.is_malformed_arguments:
                            self.get_logger().warning(
                                f'[{turn_id}] Rejected malformed tool arguments: {proposal.action_name}'
                            )
                        elif proposal.rejection_reason == "compound_task_must_stay_atomic":
                            self.get_logger().warning(
                                f'[{turn_id}] Rejected split compound-task tool: {proposal.action_name}'
                            )
                        else:
                            self.get_logger().warning(
                                f'[{turn_id}] Rejected tool proposal: '
                                f'name={proposal.action_name} reason={proposal.rejection_reason}'
                            )
                        if proposal.should_continue:
                            continue
                        break

                    # Dispatch to dedicated camera photo workflow.
                    # 转入专用拍照存盘工作流。
                    if proposal.is_camera_photo:
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_photo(turn_id, user_prompt)
                        return

                    # Dispatch to dedicated camera inspection workflow.
                    # 转入专用视觉观察检测工作流。
                    if proposal.is_camera_inspection:
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_inspection(turn_id, user_prompt)
                        return

                    # Dispatch to dedicated conditional task workflow.
                    # 转入专用复合条件任务工作流。
                    if proposal.is_conditional_task:
                        self.get_logger().info(
                            f'[{turn_id}] Conditional task workflow requested.'
                        )
                        self._process_conditional_task(
                            turn_id,
                            user_prompt,
                            proposal.plan or proposal.arguments,
                        )
                        return

                    # Record pending standard physical action.
                    # 记录待执行的标准实体动作。
                    action_payload = state.record_pending_action(
                        turn_id,
                        proposal.action_name,
                        proposal.arguments,
                    )
                    self.get_logger().info(f'[{turn_id}] Tool call: {action_payload["name"]}')

                if decision.finish_reason:
                    finish_reason = decision.finish_reason
                    # rclpy identifies a logging call by its source location. Calling
                    # different severity methods through one ``log`` variable makes a
                    # later request fail when its finish reason changes.
                    # rclpy 依据源码调用位置定位日志点，因此分开调用 info / warning。
                    if finish_reason == 'length':
                        self.get_logger().warning(
                            f'[{turn_id}] LLM stream completed: finish_reason={finish_reason}'
                        )
                    else:
                        self.get_logger().info(
                            f'[{turn_id}] LLM stream completed: finish_reason={finish_reason}'
                        )

        except Exception as e:
            self.get_logger().error(f'[{turn_id}] LLM request/stream failed: {e}\n{traceback.format_exc()}')
            if not state.corrected_text_published:
                publish_corrected(user_prompt)
            failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
            publish_spoken(failure_text)
            self._publish_screen_dialog(turn_id, state.corrected_text or user_prompt, failure_text, state.actions, error=str(e))
            self._finish_tts_turn(turn_id)
            return

        # Execute physical action sequence (via Behavior Tree or legacy direct dispatch).
        # 执行物理动作序列（优先通过行为树调度，或平滑回退直接派发）。
        if state.pending_actions:
            sequence = self._execute_dialog_action_sequence(
                state.pending_actions,
                user_prompt=user_prompt,
                turn_id=turn_id,
            )
            state.apply_action_sequence(sequence, turn_id)

        # Emit tail TTS clause if punctuation did not close it during streaming.
        # 若流式末尾存在未被标点闭合的剩余文本从句，进行最终播报。
        clean_tail = state.take_tail_tts()
        if clean_tail:
            publish_spoken(clean_tail)

        # Decide final turn outcome (check if empty-answer retry is required).
        # 计算最终回合输出（检查是否需要触发空回复单次重试）。
        decision = state.decide_final_turn()
        clean_text = decision.clean_text
        if decision.needs_empty_answer_retry:
            clean_text = state.resolve_final_text(
                self._retry_empty_answer(
                    turn_id,
                    state.corrected_text or user_prompt,
                )
            )

        if decision.should_publish_final_tts:
            publish_spoken(clean_text)

        # Publish complete reply text.
        # 发布模型完整纯文本回复。
        if clean_text:
            full_msg = String()
            full_msg.data = clean_text
            self.full_ai_publisher.publish(full_msg)

        # Persist conversation turn into history.
        # 记录本轮对话到历史队列。
        LLMConversationHistory.record_turn(
            self.chat_history,
            user_text=decision.user_text,
            assistant_text=clean_text,
            actions=state.actions,
            turn_id=turn_id,
        )

        # Atomic screen dialogue update with executed actions.
        # 原子发布屏幕对话内容与已执行动作。
        self._publish_screen_dialog(turn_id, decision.user_text, clean_text, state.actions)

        # TTS 和播放节点会按顺序处理该标记；真正播完后再恢复 ASR。
        # Audio node processes this mark in-order; ASR is resumed only after playback finishes.
        self._finish_tts_turn(turn_id)

    def _execute_dialog_action(self, name, arguments, turn_id):
        """Wait on the worker so consecutive proposals cannot interrupt each other.

        执行单个对话物理动作：在工作线程中阻塞等待执行回执，防止连续动作互相冲撞打断。
        """
        def publish(payload):
            command = json.loads(payload)
            command['turn_id'] = turn_id
            self.action_publisher.publish(String(data=json.dumps(command, ensure_ascii=False)))

        return self._action_executor.execute(
            name,
            arguments,
            publish=publish,
            owner_available=lambda: self.action_publisher.get_subscription_count() > 0,
            timeout=20.0,
            source='llm_dialog',
            cancelled=lambda: (
                not getattr(self, '_worker_running', True)
                or getattr(self, '_game_mode', 'robot') != 'robot'
            ),
        )

    def _execute_dialog_action_sequence(self, actions, *, user_prompt, turn_id):
        """Prefer the native tree owner and safely fall back when it is absent.

        执行对话动作序列：优先委托给原生行为树执行；若行为树不可用则安全降级。
        """
        return self._action_plan_workflow().invoke(
            turn_id=turn_id,
            user_prompt=user_prompt,
            actions=actions,
        )

    def _action_plan_workflow(self):
        """Lazy initializer for LLMActionPlanWorkflow.

        懒加载初始化动作计划工作流对象。
        """
        workflow = getattr(self, '_llm_action_plan_workflow', None)
        if workflow is None:
            cancelled = lambda: (
                not getattr(self, '_worker_running', True)
                or getattr(self, '_game_mode', 'robot') != 'robot'
            )
            workflow = LLMActionPlanWorkflow(
                native_available=lambda: (
                    getattr(self, '_behavior_tree_executor', None) is not None
                ),
                authorize=validate_action_call,
                execute_plan=lambda plan: self._try_execute_native_plan(
                    plan, cancelled=cancelled
                ),
                execute_action=lambda name, arguments, turn_id: (
                    self._execute_dialog_action(name, arguments, turn_id)
                ),
                cancelled=cancelled,
            )
            self._llm_action_plan_workflow = workflow
        return workflow

    def _try_execute_native_plan(self, plan, *, timeout=None, cancelled=None):
        """Attempt to execute a plan via the native plan adapter.

        尝试通过行为树适配器执行原生动作计划。
        """
        return self._native_plan().try_execute(
            plan, timeout=timeout, cancelled=cancelled
        )

    def _native_plan(self):
        """Lazy initializer for NativePlanExecutionAdapter.

        懒加载初始化原生计划执行适配器。
        """
        adapter = getattr(self, '_native_plan_adapter', None)
        if adapter is None:
            executor = getattr(self, '_behavior_tree_executor', None)
            publisher = getattr(self, 'behavior_tree_publisher', None)
            cancel_publisher = getattr(self, 'behavior_tree_cancel_publisher', None)
            adapter = NativePlanExecutionAdapter(
                ros2_executor=getattr(self, '_ros2_task_executor', None),
                legacy_executor=executor,
                publish=(lambda payload: publisher.publish(String(data=payload)))
                if publisher is not None else None,
                cancel_publish=(
                    lambda payload: cancel_publisher.publish(String(data=payload))
                ) if cancel_publisher is not None else None,
                owner_available=(
                    lambda: publisher is not None and publisher.get_subscription_count() > 0
                ) if publisher is not None else None,
            )
            self._native_plan_adapter = adapter
        return adapter

    def _on_behavior_tree_status(self, message):
        """Receive asynchronous status updates from behavior tree execution.

        接收行为树异步执行状态通知的回调函数。
        """
        self._native_plan().accept_status(message.data)

    def _process_deterministic_safety_action(
        self,
        turn_id,
        user_prompt,
        action_name,
        action_arguments,
    ):
        """Dispatch an explicit stop locally without relying on model output.

        确定性安全动作处理：在本地直接派发急停指令，无需依赖大模型推理，确保即时生效。
        """
        action_payload = {
            'turn_id': turn_id,
            'name': action_name,
            'arguments': json.dumps(action_arguments, ensure_ascii=False),
        }
        action_msg = String()
        action_msg.data = json.dumps(action_payload, ensure_ascii=False)
        self.action_publisher.publish(action_msg)
        self.get_logger().info(
            f'[{turn_id}] Deterministic safety action: {action_name} {action_arguments}'
        )

        self.corrected_publisher.publish(String(data=user_prompt))
        acknowledgement = action_acknowledgement([action_payload])
        self._publish_tts(acknowledgement, turn_id)
        LLMConversationHistory.record_dialog_turn(
            self.chat_history,
            user_text=user_prompt,
            assistant_text=acknowledgement,
        )
        self.full_ai_publisher.publish(String(data=acknowledgement))
        self._publish_screen_dialog(
            turn_id,
            user_prompt,
            acknowledgement,
            [action_payload],
        )
        self._finish_tts_turn(turn_id)

    def _process_camera_inspection(self, turn_id, user_prompt):
        """Run the camera graph and publish exactly one user-facing answer.

        执行摄像头检测工作流：启动实时预览捕获末帧，送入视觉模型分析，发布单次用户答复。
        """
        workflow = getattr(self, '_camera_inspection_workflow', None)
        if workflow is None:
            workflow = CameraInspectionWorkflow(
                capture=lambda: self._run_camera_preview(
                    duration_ms=self.tft_preview_settings.recognition_duration_ms,
                ),
                analyze=self._analyze_camera_frame,
            )
            self._camera_inspection_workflow = workflow
        result = workflow.invoke(turn_id=turn_id, user_prompt=user_prompt)
        answer = result['answer']
        error = result.get('error')

        if error:
            self.get_logger().warning(
                f'[{turn_id}] Camera inspection completed with error: {error}'
            )
        else:
            LLMConversationHistory.record_dialog_turn(
                self.chat_history,
                user_text=user_prompt,
                assistant_text=answer,
            )
            self.full_ai_publisher.publish(String(data=answer))

        self._publish_tts(answer, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, answer, [], error=error)
        self._finish_tts_turn(turn_id)

    def _on_action_status(self, message):
        """Receive asynchronous action execution status from action coordinator.

        接收来自动作协调器的动作执行状态回调函数。
        """
        executor = getattr(self, '_action_executor', None)
        if executor is not None:
            executor.accept_status(message.data)

    def _process_conditional_task_request(self, turn_id, user_prompt):
        """Force a compound intent through the plan tool or fail closed.

        处理条件任务规划请求：强制要求大模型输出条件任务计划工具调用，失败时安全闭环。
        """
        self.corrected_publisher.publish(String(data=user_prompt))
        try:
            for data in self.llm.chat_stream(
                user_prompt,
                self._history_for_request(),
                tools_enabled=True,
                only_action_name=CONDITIONAL_TASK_TOOL_NAME,
                max_tokens_override=512,
            ):
                decision = evaluate_native_conditional_event(
                    data,
                    user_prompt=user_prompt,
                )
                if decision.is_ignored or decision.is_malformed:
                    continue
                if decision.is_rejected:
                    self.get_logger().warning(
                        f'[{turn_id}] Rejected conditional plan: {decision.rejection_reason}; '
                        f'plan={json.dumps(decision.plan, ensure_ascii=False)}'
                    )
                    continue
                if decision.is_accepted and decision.plan is not None:
                    self._process_conditional_task(turn_id, user_prompt, decision.plan)
                    return

            fallback_plan = self._plan_conditional_task_as_json(user_prompt)
            decision = evaluate_conditional_plan(
                fallback_plan,
                user_prompt=user_prompt,
            )
            if decision.is_accepted and decision.plan is not None:
                self._process_conditional_task(turn_id, user_prompt, decision.plan)
                return
            error = decision.rejection_reason or 'conditional_plan_missing'
        except Exception as exc:
            error = str(exc)
            self.get_logger().error(
                f'[{turn_id}] Conditional planning failed: {exc}\n{traceback.format_exc()}'
            )

        answer = '这个条件任务没有生成可执行计划，所以我没有观察或执行动作。'
        LLMConversationHistory.record_dialog_turn(
            self.chat_history,
            user_text=user_prompt,
            assistant_text=answer,
        )
        self.full_ai_publisher.publish(String(data=answer))
        self._publish_tts(answer, turn_id)
        self._publish_screen_dialog(
            turn_id,
            user_prompt,
            answer,
            [],
            error=error,
        )
        self._finish_tts_turn(turn_id)

    def _plan_conditional_task_as_json(self, user_prompt):
        """Compatibility fallback for providers that omit function calls.

        兼容性兜底：当模型服务商不支持原生工具调用时，通过结构化 Prompt 提取条件任务 JSON。
        """
        request = build_conditional_fallback_request(user_prompt)
        chunks = []
        for data in self.llm.chat_stream(
            request.prompt,
            request.history,
            tools_enabled=request.tools_enabled,
            structured_answer=request.structured_answer,
            system_prompt=request.system_prompt,
            max_tokens_override=request.max_tokens_override,
        ):
            if data.get('type') == 'text' and data.get('content'):
                chunks.append(data['content'])
        return parse_conditional_fallback_json(chunks)

    def _process_conditional_task(self, turn_id, user_prompt, plan):
        """Run one validated observe-condition-action graph to completion.

        执行已校验的条件任务图：协调相机观察、视觉条件判断与分支动作执行。
        """
        workflow = getattr(self, '_conditional_task_workflow', None)
        if workflow is None:
            workflow = ConditionalTaskWorkflow(
                capture=lambda: self._run_camera_preview(
                    duration_ms=self.tft_preview_settings.recognition_duration_ms,
                ),
                evaluate=self._evaluate_camera_condition,
                authorize=lambda _prompt, name, arguments: validate_action_arguments(
                    name, arguments
                ),
                execute=self._execute_workflow_action,
            )
            self._conditional_task_workflow = workflow

        try:
            result = workflow.invoke(
                turn_id=turn_id,
                user_prompt=user_prompt,
                plan=plan,
            )
            outcome = build_conditional_task_outcome(result, plan)
        except Exception as exc:
            self.get_logger().error(
                f'[{turn_id}] Conditional task failed: {exc}\n{traceback.format_exc()}'
            )
            outcome = build_conditional_task_failure_outcome(str(exc))

        if outcome.error:
            self.get_logger().warning(
                f'[{turn_id}] Conditional task completed with error: {outcome.error}'
            )
        else:
            self.get_logger().info(f'[{turn_id}] Conditional task completed.')

        LLMConversationHistory.record_dialog_turn(
            self.chat_history,
            user_text=user_prompt,
            assistant_text=outcome.answer,
        )
        self.full_ai_publisher.publish(String(data=outcome.answer))
        self._publish_tts(outcome.answer, turn_id)
        self._publish_screen_dialog(
            turn_id,
            user_prompt,
            outcome.answer,
            outcome.actions,
            error=outcome.error,
        )
        self._finish_tts_turn(turn_id)

    def _evaluate_camera_condition(self, frame, observation, condition):
        """Ask the visual model for a closed yes/no/uncertain decision.

        请求视觉多模态大模型对画面中的指定视觉条件做封闭式二分类判定（yes/no/uncertain）。
        """
        request = build_conditional_vision_request(
            frame,
            observation=observation,
            condition=condition,
        )
        accumulator = VisualResponseAccumulator()
        for data in self.llm.chat_stream(
            request.prompt,
            request.history,
            image_base64=request.image_base64,
            tools_enabled=request.tools_enabled,
            structured_answer=request.structured_answer,
            system_prompt=request.system_prompt,
            max_tokens_override=request.max_tokens_override,
        ):
            accumulator.process_event(data)
        return accumulator.raw_text()

    def _execute_workflow_action(self, name, arguments):
        """Execute a physical action triggered by the conditional task workflow.

        执行条件任务工作流中触发的具体物理动作指令。
        """
        executor = self._action_executor
        return executor.execute(
            name,
            arguments,
            publish=lambda payload: self.action_publisher.publish(String(data=payload)),
            owner_available=lambda: self.action_publisher.get_subscription_count() > 0,
            timeout=20.0,
            source='llm_conditional_task',
        )

    def _analyze_camera_frame(self, frame, user_prompt):
        """Return a plain-text visual answer without requiring tool calling.

        将相机捕获的画面帧和用户提问送给视觉模型，返回纯文本分析结果（不需要工具调用）。
        """
        request = build_camera_qa_request(
            frame,
            user_prompt=user_prompt,
            history=self._visual_history(),
        )
        accumulator = VisualResponseAccumulator()
        for data in self.llm.chat_stream(
            request.prompt,
            request.history,
            image_base64=request.image_base64,
            tools_enabled=request.tools_enabled,
            structured_answer=request.structured_answer,
            system_prompt=request.system_prompt,
            max_tokens_override=request.max_tokens_override,
        ):
            accumulator.process_event(data)
        return accumulator.clean_answer()

    def _process_camera_photo(self, turn_id, user_prompt):
        """Confirm -> TFT preview for configured duration -> save final frame to disk.

        拍照流程：即时语音确认 -> 屏幕预览指定时长 -> 保存最后一帧至本地硬盘（不调用视觉模型）。
        """
        confirmation = '好的，准备拍照。'
        self._publish_tts(confirmation, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, confirmation, [])

        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.photo_duration_ms,
        )
        decision = decide_photo_preview(preview)
        if decision.should_save:
            try:
                photo_path = save_camera_photo(
                    decision.frame,
                    self.tft_preview_settings.photo_directory,
                )
                self.get_logger().info(f'[{turn_id}] Photo saved: {photo_path}')
                decision = photo_save_succeeded()
            except Exception as exc:
                self.get_logger().error(
                    f'[{turn_id}] Photo save failed: {exc}\n{traceback.format_exc()}'
                )
                decision = photo_save_failed(str(exc))

        self._publish_tts(decision.answer, turn_id)
        LLMConversationHistory.record_dialog_turn(
            self.chat_history,
            user_text=user_prompt,
            assistant_text=decision.answer,
        )
        self.full_ai_publisher.publish(String(data=decision.answer))
        self._publish_screen_dialog(turn_id, user_prompt, decision.answer, [], error=decision.error)
        self._finish_tts_turn(turn_id)

    def _run_camera_preview(self, *, duration_ms):
        """Run camera capture on the LLM worker, never on the ROS callback thread.

        在后台工作线程中启动相机预览租约，避免占用 ROS 回调线程。
        """
        return self.tft_preview.send_camera_preview(
            duration_ms=duration_ms,
            hold_ms=self.tft_preview_settings.hold_ms,
            fps=self.tft_preview_settings.fps,
        )

    def _visual_history(self):
        """Build text-only dialogue history for multimodal visual requests.

        为视觉多模态请求构建近期的纯文本对话历史。
        """
        return LLMConversationHistory.build_visual_history(self.chat_history)

    def _history_for_request(self):
        """Build request history starting with a user turn and stripping non-text blocks.

        构建标准请求对话历史：以 user 角色起始，剥离图片块并限制最大条数。
        """
        max_messages = getattr(self, 'CHAT_HISTORY_MESSAGES', 12)
        return LLMConversationHistory.build_request_history(
            self.chat_history, max_messages=max_messages
        )

    @staticmethod
    def _text_only_history_message(item):
        """Compatibility helper: convert history entry to text-only structure.

        兼容辅助方法：将单条历史消息转换为纯文本消息格式。
        """
        return LLMConversationHistory.text_only_history_message(item)

    @classmethod
    def _rejected_action_reply(cls, rejected_actions):
        """Compatibility helper: build user explanation when an action is rejected.

        兼容辅助方法：当动作被拦截拒绝时生成告知用户的解释文案。
        """
        return StreamResponseAccumulator.rejected_action_reply(rejected_actions)

    @classmethod
    def _clean_visual_answer(cls, text):
        """Compatibility helper: clean model text response for visual QA.

        兼容辅助方法：清洗视觉模型答复文本。
        """
        return LLMResponsePolicy.clean_visual_answer(text)

    def _publish_tts(self, text, turn_id=''):
        """Sanitize text and publish to TTS topic.

        清洗文本并发布到 TTS 语音合成话题。
        """
        safe = self._sanitize_speech_text(text)
        if not safe:
            return ''
        self.tts_publisher.publish(String(data=safe))
        self.get_logger().info(f'[{turn_id}] Published TTS sentence: {safe}')
        return safe

    def _finish_tts_turn(self, turn_id):
        """Publish turn-end delimiter for audio playback node lifecycle.

        发布回合结束定界符，通知音频播放节点本轮语音已队列完毕，播完后可恢复 ASR。
        """
        self.tts_publisher.publish(String(data=encode_turn_end(turn_id)))
        self.get_logger().info(f'[{turn_id}] TTS turn queued; waiting for playback completion.')

    def _retry_empty_answer(self, turn_id, user_prompt):
        """Retry once without tools when the model returned only the ASR correction line.

        当大模型仅返回 ASR 纠错行而无正文答复时，关闭工具免重试一次。
        """
        self.get_logger().warning(
            f'[{turn_id}] LLM returned no answer text; retrying once without tools.'
        )
        try:
            request = build_empty_answer_retry_request(
                user_prompt,
                history=self._history_for_request(),
                settings=getattr(self.llm, 'settings', {}),
            )
            accumulator = EmptyAnswerRetryAccumulator()
            for data in self.llm.chat_stream(
                request.prompt,
                request.history,
                tools_enabled=request.tools_enabled,
                structured_answer=request.structured_answer,
                max_tokens_override=request.max_tokens_override,
            ):
                accumulator.process_event(data)
            answer = accumulator.parse_answer()
            if answer:
                self.get_logger().info(f'[{turn_id}] Empty-answer retry succeeded.')
            return answer
        except Exception as exc:
            self.get_logger().error(
                f'[{turn_id}] Empty-answer retry failed: {exc}\n{traceback.format_exc()}'
            )
            return ''

    @classmethod
    def _is_long_form_request(cls, user_prompt):
        """Compatibility helper: check whether user prompt is a long-form request.

        兼容辅助方法：检测用户输入是否属于长文生成请求。
        """
        return LLMResponsePolicy.is_long_form_request(user_prompt)

    @classmethod
    def _needs_action_tools(cls, user_prompt):
        """Compatibility helper: ordinary dialogue always receives control tools.

        Dedicated visual inspection and empty-answer retry paths explicitly
        disable tools at their call sites, rather than relying on wording.

        兼容辅助方法：普通对话始终启用控制工具。
        """
        del user_prompt
        return True

    def _max_tokens_for_request(self, is_long_form):
        """Compatibility helper: compute max token budget for dialogue turn.

        兼容辅助方法：根据长文标记计算 token 预算上限。
        """
        settings = getattr(self.llm, 'settings', {})
        configured_tokens = settings.get('max_tokens', 0) if isinstance(settings, dict) else 0
        return LLMResponsePolicy.max_tokens(configured_tokens, long_form=is_long_form)

    @classmethod
    def _extract_corrected_text(cls, first_line):
        """Compatibility helper: extract corrected text from ASR correction prefix.

        兼容辅助方法：从首行纠错前缀中提取纠错后的文本。
        """
        return LLMResponsePolicy.extract_corrected_text(first_line)

    @classmethod
    def _strip_correction_line(cls, text):
        """Compatibility helper: strip ASR correction line from answer.

        兼容辅助方法：剔除模型答复中的首行 ASR 纠错标记。
        """
        return LLMResponsePolicy.strip_correction_line(text)

    @classmethod
    def _sanitize_speech_text(cls, text):
        """Compatibility helper: sanitize speech text for TTS synthesis.

        兼容辅助方法：清洗用于 TTS 朗读的纯净文本。
        """
        return LLMResponsePolicy.sanitize_speech_text(text)

    @classmethod
    def _is_correction_label_only(cls, text):
        """Compatibility helper: check if text is only a correction label.

        兼容辅助方法：检测文本是否仅为纠错标签而无内容。
        """
        return LLMResponsePolicy.is_correction_label_only(text)

    @classmethod
    def _strip_answer_prefix(cls, text):
        """Compatibility helper: strip '回答：' prefix from model output.

        兼容辅助方法：剥离模型输出中的 '回答：' 前缀。
        """
        return LLMResponsePolicy.strip_answer_prefix(text)

    def _publish_screen_dialog(self, turn_id, corrected_text, ai_text, actions, error=None):
        """Publish atomic JSON message for the robot screen dialogue UI.

        原子发布屏幕对话 UI 消息（包含当前回合识别文本、回答正文与执行动作）。
        """
        payload = {
            'turn_id': turn_id,
            'corrected_text': corrected_text or '',
            'ai_text': ai_text or '',
            'actions': actions or [],
        }
        if error:
            payload['error'] = error

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.screen_dialog_publisher.publish(msg)
        self.get_logger().info(f'[{turn_id}] Published atomic screen dialog.')

    def destroy_node(self):
        """Gracefully stop worker thread, close preview connection, and destroy node.

        优雅关闭节点：停止工作线程循环、关闭 TFT 预览客户端连接并销毁 ROS 节点。
        """
        self._worker_running = False
        if hasattr(self, '_request_queue'):
            try:
                self._request_queue.put_nowait(None)
            except queue.Full:
                pass
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if getattr(self, 'tft_preview', None) is not None:
            self.tft_preview.close()
        super().destroy_node()


def main(args=None):
    """Node entry point initializing rclpy and spinning LLMBrainNode.

    节点主入口：初始化 rclpy 并运行 LLMBrainNode 节点事件循环。
    """
    rclpy.init(args=args)
    node = LLMBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
