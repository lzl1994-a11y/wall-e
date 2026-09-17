#!/usr/bin/env python3
import base64
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
from services.llm_conditional_planning import (
    build_conditional_fallback_request,
    evaluate_conditional_plan,
    evaluate_native_conditional_event,
    parse_conditional_fallback_json,
)
from services.action_status import ACTION_STATUS_TOPIC
from services.llm_service import LLMService
from services.llm_response_policy import LLMResponsePolicy
from services.llm_stream_response import StreamResponseAccumulator
from services.llm_conversation_history import LLMConversationHistory
from services.llm_request_preparation import (
    ROUTE_CAMERA_INSPECTION,
    ROUTE_CAMERA_PHOTO,
    ROUTE_CONDITIONAL_TASK,
    ROUTE_SAFETY_ACTION,
    prepare_voice_request,
)
from services.llm_tool_proposal import evaluate_tool_proposal
from services.camera_frame import save_camera_photo
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_STATE_TOPIC,
    decode_game_frame,
    game_mode_from_message,
)
from services.game_tft_stream import prepare_game_bgr
from services.game_commentary import GameCommentaryController
from services.tft_preview_client import TftPreviewClient
from services.tft_preview_server import load_tft_preview_settings
from services.tts_protocol import encode_turn_end
from services.dialog_expression_protocol import (
    DIALOG_EXPRESSION_TOPIC,
    encode_dialog_expression,
)
from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.llm_action_plan import LLMActionPlanWorkflow
from services.dialog_workflow import CameraInspectionWorkflow, ConditionalTaskWorkflow
from services.voice_debug import RollingVoiceDebugStore


class LLMBrainNode(Node):
    CHAT_HISTORY_MESSAGES = 12
    LONG_FORM_REQUEST_RE = LLMResponsePolicy.LONG_FORM_REQUEST_RE
    LONG_FORM_MAX_TOKENS = LLMResponsePolicy.LONG_FORM_MAX_TOKENS
    FIRST_TTS_CLAUSE_MIN_CHARS = 10
    CLAUSE_PUNCTUATIONS = {'，', ',', '；', ';', '：', ':'}
    CORRECTION_LABELS = LLMResponsePolicy.CORRECTION_LABELS
    TTS_CLEAN_RE = LLMResponsePolicy.TTS_CLEAN_RE
    OUTPUT_LINE_PREFIX_RE = LLMResponsePolicy.OUTPUT_LINE_PREFIX_RE

    def __init__(self):
        super().__init__('walle_llm_brain')

        self.llm = None
        self._camera_inspection_workflow = None
        self._conditional_task_workflow = None
        self._llm_action_plan_workflow = None
        self._action_executor = CorrelatedActionExecutor()
        self._behavior_tree_executor = CorrelatedPlanExecutor()
        self._ros2_task_executor = None
        self._voice_debug = RollingVoiceDebugStore()
        self.chat_history = deque(maxlen=24)
        self.punctuations = {'。', '？', '.', '?', '！', '!'}
        self._request_queue = queue.Queue(maxsize=8)
        self._worker_running = False
        self._game_mode = "robot"
        self._game_commentary = GameCommentaryController()

        # Create ROS endpoints before the slow LLM client init. This lets DDS
        # discover `voice_text` while the model service is warming up.
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
            action_client, goal_type = create_wali_task_action_client(
                self, Path(__file__).resolve().parent.parent
            )
            self._ros2_task_executor = Ros2ActionPlanExecutor(action_client, goal_type)
            self.get_logger().info('BehaviorTree ROS2 Action client is ready.')
        except Exception as exc:
            self.get_logger().warning(
                f'BehaviorTree ROS2 Action client unavailable; legacy fallback remains active: {exc}'
            )
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
        self.create_subscription(
            String,
            ACTION_STATUS_TOPIC,
            self._on_action_status,
            10,
        )
        self.corrected_publisher = self.create_publisher(String, 'corrected_text', 10)
        self.full_ai_publisher = self.create_publisher(String, 'full_ai_text', 10)
        self.screen_dialog_publisher = self.create_publisher(String, 'screen_dialog', 10)
        self.busy_publisher = self.create_publisher(String, 'llm_busy', 10)
        self.dialog_expression_publisher = self.create_publisher(
            String, DIALOG_EXPRESSION_TOPIC, 10
        )
        self.tft_preview_settings = load_tft_preview_settings()
        self.tft_preview = TftPreviewClient(self, logger=self.get_logger())
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_timer(1.0, self._game_commentary_tick)

        try:
            self.llm = LLMService()
            self.get_logger().info('LLM service initialized.')
        except Exception as e:
            self.get_logger().error(f'LLM service initialization failed: {e}')
            return

        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._llm_worker,
            name='llm-worker',
            daemon=True,
        )
        self._worker_thread.start()

    def _on_game_state(self, message):
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
        frame = decode_game_frame(bytes(message.data))
        if frame is None:
            return
        self._game_commentary.accept_frame(frame)

    def _game_commentary_tick(self):
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
        """Queue the request so the ROS callback thread is never blocked by LLM I/O."""
        if self._game_mode != "robot":
            self.get_logger().info("游戏模式热备中，忽略语音 LLM 输入")
            return
        user_prompt = (msg.data or '').strip()
        # 过滤掉常见的 ASR 噪声音译（如 #，或者单纯的标点符号）
        if not user_prompt or user_prompt == '#' or len(user_prompt.strip('.,?!。，？！# ')) == 0:
            self.get_logger().info(f'Ignored empty/noise ASR input: "{user_prompt}"')
            return

        turn_id = uuid.uuid4().hex[:12]
        self.get_logger().info(f'[{turn_id}] Voice text received: {user_prompt}')

        # Show what was heard immediately. Waiting for LLM/tool completion made
        # the user's text appear frozen during a slow model response.
        self._publish_screen_dialog(turn_id, user_prompt, '', [])

        try:
            self._request_queue.put_nowait({
                'turn_id': turn_id,
                'user_prompt': user_prompt,
            })
        except queue.Full:
            self.get_logger().error('LLM request queue is full; dropped this voice input.')

    def _llm_worker(self):
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
        if self._game_mode != "playing":
            return
        turn_id = task['turn_id']
        self.busy_publisher.publish(String(data="busy"))
        prompt = (
            "观察这张正在运行的 FC 游戏画面，以瓦力的口吻说一句简短自然的中文评论。"
            "可以提醒危险、鼓励玩家或描述关键局面；看不清时不要猜。只输出可直接播报的一句话。"
        )
        try:
            chunks = []
            for data in self.llm.chat_stream(
                prompt,
                [],
                image_base64=base64.b64encode(task['jpeg']).decode('ascii'),
                tools_enabled=False,
                structured_answer=False,
                system_prompt=(
                    "你是陪主人玩 FC 游戏的瓦力。只依据当前游戏截图简短评论，"
                    "不输出分析过程。"
                ),
                max_tokens_override=96,
            ):
                if data.get('type') == 'text' and data.get('content'):
                    chunks.append(data['content'])
            answer = self._clean_visual_answer(''.join(chunks))
            if not answer:
                raise RuntimeError('游戏视觉模型返回空答案')
            self._publish_tts(answer, turn_id)
            self.full_ai_publisher.publish(String(data=answer))
        except Exception as exc:
            self.get_logger().error(f'[{turn_id}] Game vision failed: {exc}')
        finally:
            self._finish_tts_turn(turn_id)

    def _process_voice_task(self, turn_id, user_prompt):
        # 通知 STT 节点暂停 ASR
        busy_msg = String()
        busy_msg.data = "busy"
        self.busy_publisher.publish(busy_msg)

        model_settings = getattr(self.llm, 'settings', {})
        prepared_request = prepare_voice_request(
            user_prompt,
            model_settings=model_settings,
        )

        if prepared_request.route == ROUTE_SAFETY_ACTION:
            self._process_deterministic_safety_action(
                turn_id,
                user_prompt,
                prepared_request.safety_action_name,
                prepared_request.safety_action_arguments,
            )
            return

        if prepared_request.route == ROUTE_CONDITIONAL_TASK:
            self._process_conditional_task_request(turn_id, user_prompt)
            return

        if prepared_request.route == ROUTE_CAMERA_PHOTO:
            self._process_camera_photo(turn_id, user_prompt)
            return

        if prepared_request.route == ROUTE_CAMERA_INSPECTION:
            self._process_camera_inspection(turn_id, user_prompt)
            return

        augmented_prompt = prepared_request.augmented_prompt
        tools_enabled = prepared_request.tools_enabled
        max_tokens_override = prepared_request.max_tokens_override

        self.get_logger().info(f'[{turn_id}] Sending request to LLM...')
        self.get_logger().info(f'[{turn_id}] Control tools enabled for semantic handling.')
        if max_tokens_override is not None:
            self.get_logger().info(
                f'[{turn_id}] Long-form request detected; max_tokens={max_tokens_override}.'
            )

        accumulator = StreamResponseAccumulator(
            punctuations=getattr(self, 'punctuations', {'。', '？', '.', '?', '！', '!'}),
            clause_punctuations=getattr(self, 'CLAUSE_PUNCTUATIONS', {'，', ',', '；', ';', '：', ':'}),
            first_tts_clause_min_chars=getattr(self, 'FIRST_TTS_CLAUSE_MIN_CHARS', 10),
        )
        corrected_text = ''
        corrected_text_published = False
        actions = []
        pending_actions = []
        action_failure = None
        expression_published = False

        def publish_corrected(value):
            nonlocal corrected_text, corrected_text_published
            corrected_text = (value or user_prompt).strip() or user_prompt
            corrected_text_published = True
            msg = String()
            msg.data = corrected_text
            self.corrected_publisher.publish(msg)
            self.get_logger().info(
                f'[{turn_id}] Corrected text: raw="{user_prompt}" corrected="{corrected_text}"'
            )

        def publish_spoken(value):
            nonlocal expression_published
            if not expression_published:
                expression_publisher = getattr(
                    self, "dialog_expression_publisher", None
                )
                if expression_publisher is not None:
                    expression_publisher.publish(String(
                        data=encode_dialog_expression("neutral", "low", turn_id)
                    ))
                expression_published = True
            spoken = self._publish_tts(value, turn_id)
            if spoken:
                accumulator.record_spoken(spoken)

        # Correction metadata is an internal concern. Publish the ASR text for
        # the existing topic contract and ask the model for speech only.
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
                decision = accumulator.process_event(data)

                for tts_text in decision.tts_sentences:
                    publish_spoken(tts_text)

                if decision.expression:
                    self.dialog_expression_publisher.publish(String(
                        data=encode_dialog_expression(
                            decision.expression.get('expression'),
                            decision.expression.get('intensity'),
                            turn_id,
                        )
                    ))
                    expression_published = True

                if decision.tool_call:
                    proposal = evaluate_tool_proposal(
                        decision.tool_call,
                        user_prompt=user_prompt,
                        conditional_request=prepared_request.is_conditional_task,
                    )
                    if proposal.is_rejected:
                        accumulator.record_rejected_action(
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

                    if proposal.is_camera_photo:
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_photo(turn_id, user_prompt)
                        return

                    if proposal.is_camera_inspection:
                        self.get_logger().info(f'[{turn_id}] Camera inspection tool requested.')
                        self._process_camera_inspection(turn_id, user_prompt)
                        return

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

                    action_payload = {
                        'turn_id': turn_id,
                        'name': proposal.action_name,
                        'arguments': json.dumps(proposal.arguments, ensure_ascii=False),
                    }
                    actions.append(action_payload)
                    pending_actions.append({
                        'name': proposal.action_name,
                        'arguments': proposal.arguments,
                    })
                    accumulator.record_tool_proposal(action_payload)
                    self.get_logger().info(f'[{turn_id}] Tool call: {action_payload["name"]}')

                if decision.finish_reason:
                    finish_reason = decision.finish_reason
                    # rclpy identifies a logging call by its source location.  Calling
                    # different severity methods through one ``log`` variable makes a
                    # later request fail when its finish reason changes.
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
            if not corrected_text_published:
                publish_corrected(user_prompt)
            failure_text = '\u6211\u521a\u624d\u5904\u7406\u5931\u8d25\u4e86\uff0c\u7a0d\u540e\u518d\u8bd5\u3002'
            publish_spoken(failure_text)
            self._publish_screen_dialog(turn_id, corrected_text or user_prompt, failure_text, actions, error=str(e))
            self._finish_tts_turn(turn_id)
            return

        if pending_actions:
            sequence = self._execute_dialog_action_sequence(
                pending_actions,
                user_prompt=user_prompt,
                turn_id=turn_id,
            )
            sequence_results = sequence.get('results', [])
            actions = []
            for result in sequence_results:
                actions.append({
                    'turn_id': turn_id,
                    'name': result.get('name') or result.get('action', ''),
                    'arguments': json.dumps(result.get('arguments', {}), ensure_ascii=False),
                    'status': result.get('status', 'failed'),
                    'request_id': result.get('request_id', ''),
                    'reason': result.get('reason', ''),
                })
            action_failure = next(
                (
                    result for result in sequence_results
                    if result.get('status') not in {'completed', 'skipped'}
                ),
                None,
            )

        clean_tail = accumulator.take_tail_tts()
        if clean_tail:
            publish_spoken(clean_tail)

        final_user_memory = corrected_text if corrected_text else user_prompt

        clean_text = accumulator.decide_final_text(
            action_failure=action_failure,
            actions=actions,
        )
        if not clean_text:
            clean_text = self._retry_empty_answer(
                turn_id,
                corrected_text or user_prompt,
            )
        if not clean_text:
            clean_text = accumulator.FALLBACK_EMPTY_REPLY
        if not accumulator.spoken_parts:
            publish_spoken(clean_text)

        if clean_text:
            full_msg = String()
            full_msg.data = clean_text
            self.full_ai_publisher.publish(full_msg)

        LLMConversationHistory.record_turn(
            self.chat_history,
            user_text=final_user_memory,
            assistant_text=clean_text,
            actions=actions,
            turn_id=turn_id,
        )

        self._publish_screen_dialog(turn_id, final_user_memory, clean_text, actions)

        # TTS 和播放节点会按顺序处理该标记；真正播完后再恢复 ASR。
        self._finish_tts_turn(turn_id)

    def _execute_dialog_action(self, name, arguments, turn_id):
        """Wait on the worker so consecutive proposals cannot interrupt each other."""
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
        """Prefer the native tree owner and safely fall back when it is absent."""
        return self._action_plan_workflow().invoke(
            turn_id=turn_id,
            user_prompt=user_prompt,
            actions=actions,
        )

    def _action_plan_workflow(self):
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
        return self._native_plan().try_execute(
            plan, timeout=timeout, cancelled=cancelled
        )

    def _native_plan(self):
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
        self._native_plan().accept_status(message.data)

    def _process_deterministic_safety_action(
        self,
        turn_id,
        user_prompt,
        action_name,
        action_arguments,
    ):
        """Dispatch an explicit stop locally without relying on model output."""
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
        """Run the camera graph and publish exactly one user-facing answer."""
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
        executor = getattr(self, '_action_executor', None)
        if executor is not None:
            executor.accept_status(message.data)

    def _process_conditional_task_request(self, turn_id, user_prompt):
        """Force a compound intent through the plan tool or fail closed."""
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
        """Compatibility fallback for providers that omit function calls."""
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
        """Run one validated observe-condition-action graph to completion."""
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
            answer = result.get('answer') or '这次任务没有完成。'
            error = result.get('error')
        except Exception as exc:
            self.get_logger().error(
                f'[{turn_id}] Conditional task failed: {exc}\n{traceback.format_exc()}'
            )
            result = {}
            answer = '这个任务计划没有通过检查，所以我没有执行动作。'
            error = str(exc)

        action_result = result.get('action_result')
        actions = []
        if isinstance(action_result, dict):
            actions.append({
                'name': action_result.get('action') or plan.get('action_name', ''),
                'arguments': json.dumps(plan.get('action_arguments', {}), ensure_ascii=False),
                'status': action_result.get('status', 'failed'),
                'request_id': action_result.get('request_id', ''),
            })
        if error:
            self.get_logger().warning(
                f'[{turn_id}] Conditional task completed with error: {error}'
            )
        else:
            self.get_logger().info(f'[{turn_id}] Conditional task completed.')

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
            actions,
            error=error,
        )
        self._finish_tts_turn(turn_id)

    def _evaluate_camera_condition(self, frame, observation, condition):
        """Ask the visual model for a closed yes/no/uncertain decision."""
        image_b64 = base64.b64encode(frame).decode('ascii')
        prompt = (
            '请只依据附带的当前摄像头画面判断条件。返回一个 JSON 对象，且只能包含 '
            'decision 和 evidence。decision 只能是 yes、no、uncertain；图片不足以确认时'
            '必须使用 uncertain。不要执行动作，不要输出 Markdown 或其他文字。\n'
            f'观察任务：{observation}\n判断条件：{condition}'
        )
        chunks = []
        for data in self.llm.chat_stream(
            prompt,
            [],
            image_base64=image_b64,
            tools_enabled=False,
            structured_answer=False,
            system_prompt=(
                '你是机器人视觉条件判断器。只能依据当前图片返回严格 JSON；'
                '无法确认时必须返回 uncertain，禁止猜测。'
            ),
            max_tokens_override=160,
        ):
            if data.get('type') == 'text' and data.get('content'):
                chunks.append(data['content'])
        return ''.join(chunks).strip()

    def _execute_workflow_action(self, name, arguments):
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
        """Return a plain-text visual answer without requiring tool calling."""
        image_b64 = base64.b64encode(frame).decode('ascii')
        visual_prompt = (
            '请根据我附带的摄像头画面回答用户的问题。只输出简短、自然、可直接播报的中文答案，'
            '不要输出修正文本标签、分析过程、工具调用或括号说明。\n'
            f'用户问题：{user_prompt}'
        )
        chunks = []
        for data in self.llm.chat_stream(
            visual_prompt,
            self._visual_history(),
            image_base64=image_b64,
            tools_enabled=False,
            structured_answer=False,
            system_prompt=(
                '你是瓦力的视觉。只依据当前摄像头图片回答问题；看不清时明确说看不清，'
                '不要猜测。答案使用简短自然的中文，不能输出分析过程或任何标签。'
            ),
        ):
            if data.get('type') == 'text' and data.get('content'):
                chunks.append(data['content'])
        return self._clean_visual_answer(''.join(chunks))

    def _process_camera_photo(self, turn_id, user_prompt):
        """确认 → TFT 预览 3 秒 → 保存末帧；不调用视觉模型。"""
        confirmation = '好的，准备拍照。'
        self._publish_tts(confirmation, turn_id)
        self._publish_screen_dialog(turn_id, user_prompt, confirmation, [])

        preview = self._run_camera_preview(
            duration_ms=self.tft_preview_settings.photo_duration_ms,
        )
        if preview.busy:
            answer = '我正在拍上一张，等一下再试。'
            error = 'camera_preview_busy'
        elif not preview.last_frame:
            answer = '这次没有拍到，检查一下摄像头连接。'
            error = preview.error or 'camera_frame_unavailable'
        else:
            try:
                photo_path = save_camera_photo(
                    preview.last_frame,
                    self.tft_preview_settings.photo_directory,
                )
                self.get_logger().info(f'[{turn_id}] Photo saved: {photo_path}')
                answer = '拍好了，照片已经保存。'
                error = None
            except Exception as exc:
                self.get_logger().error(
                    f'[{turn_id}] Photo save failed: {exc}\n{traceback.format_exc()}'
                )
                answer = '画面拍到了，但照片保存失败了。'
                error = str(exc)

        self._publish_tts(answer, turn_id)
        LLMConversationHistory.record_dialog_turn(
            self.chat_history,
            user_text=user_prompt,
            assistant_text=answer,
        )
        self.full_ai_publisher.publish(String(data=answer))
        self._publish_screen_dialog(turn_id, user_prompt, answer, [], error=error)
        self._finish_tts_turn(turn_id)

    def _run_camera_preview(self, *, duration_ms):
        """Run camera capture on the LLM worker, never on the ROS callback thread."""
        return self.tft_preview.send_camera_preview(
            duration_ms=duration_ms,
            hold_ms=self.tft_preview_settings.hold_ms,
            fps=self.tft_preview_settings.fps,
        )

    def _visual_history(self):
        return LLMConversationHistory.build_visual_history(self.chat_history)

    def _history_for_request(self):
        max_messages = getattr(self, 'CHAT_HISTORY_MESSAGES', 12)
        return LLMConversationHistory.build_request_history(
            self.chat_history, max_messages=max_messages
        )

    @staticmethod
    def _text_only_history_message(item):
        return LLMConversationHistory.text_only_history_message(item)

    @classmethod
    def _rejected_action_reply(cls, rejected_actions):
        return StreamResponseAccumulator.rejected_action_reply(rejected_actions)

    @classmethod
    def _clean_visual_answer(cls, text):
        return LLMResponsePolicy.clean_visual_answer(text)

    def _publish_tts(self, text, turn_id=''):
        safe = self._sanitize_speech_text(text)
        if not safe:
            return ''
        self.tts_publisher.publish(String(data=safe))
        self.get_logger().info(f'[{turn_id}] Published TTS sentence: {safe}')
        return safe

    def _finish_tts_turn(self, turn_id):
        self.tts_publisher.publish(String(data=encode_turn_end(turn_id)))
        self.get_logger().info(f'[{turn_id}] TTS turn queued; waiting for playback completion.')

    def _retry_empty_answer(self, turn_id, user_prompt):
        """Retry once when the model returned only the ASR correction line."""
        self.get_logger().warning(
            f'[{turn_id}] LLM returned no answer text; retrying once without tools.'
        )
        retry_prompt = (
            f'用户说：{user_prompt}\n'
            '请直接用一到两句简短自然的中文回答。只输出回答正文，不要输出纠错标签、'
            '分析过程、Markdown、动作说明或任何前缀。'
        )
        settings = getattr(self.llm, 'settings', {})
        configured_tokens = settings.get('max_tokens', 0) if isinstance(settings, dict) else 0
        retry_tokens = configured_tokens if isinstance(configured_tokens, int) else 0
        if retry_tokens <= 0:
            retry_tokens = 256
        retry_tokens = min(max(retry_tokens, 128), 256)
        try:
            chunks = []
            for data in self.llm.chat_stream(
                retry_prompt,
                self._history_for_request(),
                tools_enabled=False,
                structured_answer=False,
                max_tokens_override=retry_tokens,
            ):
                if data.get('type') == 'text' and data.get('content'):
                    chunks.append(data['content'])
            raw = ''.join(chunks).strip()
            if not raw:
                return ''
            if '\n' not in raw and self._extract_corrected_text(raw):
                return ''
            answer = self._sanitize_speech_text(raw)
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
        return LLMResponsePolicy.is_long_form_request(user_prompt)

    @classmethod
    def _needs_action_tools(cls, user_prompt):
        """Compatibility helper: ordinary dialogue always receives control tools.

        Dedicated visual inspection and empty-answer retry paths explicitly
        disable tools at their call sites, rather than relying on wording.
        """
        del user_prompt
        return True

    def _max_tokens_for_request(self, is_long_form):
        settings = getattr(self.llm, 'settings', {})
        configured_tokens = settings.get('max_tokens', 0) if isinstance(settings, dict) else 0
        return LLMResponsePolicy.max_tokens(configured_tokens, long_form=is_long_form)

    @classmethod
    def _extract_corrected_text(cls, first_line):
        return LLMResponsePolicy.extract_corrected_text(first_line)

    @classmethod
    def _strip_correction_line(cls, text):
        return LLMResponsePolicy.strip_correction_line(text)

    @classmethod
    def _sanitize_speech_text(cls, text):
        return LLMResponsePolicy.sanitize_speech_text(text)

    @classmethod
    def _is_correction_label_only(cls, text):
        return LLMResponsePolicy.is_correction_label_only(text)

    @classmethod
    def _strip_answer_prefix(cls, text):
        return LLMResponsePolicy.strip_answer_prefix(text)

    def _publish_screen_dialog(self, turn_id, corrected_text, ai_text, actions, error=None):
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

