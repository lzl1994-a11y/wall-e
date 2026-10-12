import base64
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _complete_fake_dispatcher(module):
    module.DIRECT_ANSWER_TOOL_NAME = "direct_answer"
    module.DIRECT_ANSWER_TOOL = {
        "type": "function",
        "function": {
            "name": "direct_answer",
            "description": "trusted answer",
            "parameters": {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
            },
        },
    }
    module.get_action_tools = lambda: []
    return module


class _FakeResponse:
    def __iter__(self):
        delta = types.SimpleNamespace(content="看起来是一只杯子。", tool_calls=None)
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=delta, finish_reason="stop")]
        )


class _FakeStructuredResponse:
    def __iter__(self):
        tool_call = types.SimpleNamespace(
            index=0,
            function=types.SimpleNamespace(
                name="direct_answer",
                arguments='{"response":"看起来是一只杯子。"}',
            ),
        )
        delta = types.SimpleNamespace(content=None, tool_calls=[tool_call])
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=delta, finish_reason="tool_calls")]
        )


class LLMServiceVisionTests(unittest.TestCase):
    def test_prior_image_blocks_are_removed_from_history_context(self):
        from services.llm.llm_service import _text_only_chat_history

        history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "上一轮的问题"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64,old-image",
                }},
            ],
        }]

        cleaned = _text_only_chat_history(history)

        self.assertEqual(cleaned, [{"role": "user", "content": "上一轮的问题"}])
        self.assertNotIn("old-image", repr(cleaned))

    def test_image_is_sent_as_openai_vision_content_without_answer_tool(self):
        fake_dispatcher = _complete_fake_dispatcher(
            types.ModuleType("services.llm.tool_dispatcher")
        )
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def __init__(self):
                self.calls = []

            def feed(self, delta):
                for call in getattr(delta, "tool_calls", None) or []:
                    self.calls.append({
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments),
                    })

            def flush(self):
                return self.calls

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.llm.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm.llm_service", None)
            from services.llm.llm_service import LLMService

        service = object.__new__(LLMService)
        service.settings = {
            "provider": "zhipu",
            "temperature": 0.4,
            "max_tokens": 2048,
        }
        service.system_prompt = "你是瓦力。"
        service.model = "glm-4.1v-thinking-flashx"
        service.client = MagicMock()
        service.client.chat.completions.create.return_value = _FakeResponse()

        payload = base64.b64encode(b"jpeg").decode("ascii")
        result = list(service.chat_stream(
            "这是什么？",
            image_base64=payload,
            tools_enabled=False,
            structured_answer=False,
        ))

        kwargs = service.client.chat.completions.create.call_args.kwargs
        user_message = kwargs["messages"][-1]
        self.assertIn("只给最终台词", kwargs["messages"][0]["content"])
        self.assertEqual(user_message["content"][0]["text"], "这是什么？")
        self.assertEqual(
            user_message["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64," + payload,
        )
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(result[0]["content"], "看起来是一只杯子。")
        self.assertEqual(result[-1], {"type": "done", "finish_reason": "stop"})
        sys.modules.pop("services.llm.llm_service", None)

    def test_aliyun_fast_mode_disables_thinking(self):
        fake_dispatcher = _complete_fake_dispatcher(
            types.ModuleType("services.llm.tool_dispatcher")
        )
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.llm.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm.llm_service", None)
            from services.llm.llm_service import LLMService

        service = object.__new__(LLMService)
        service.settings = {
            "provider": "aliyun",
            "reasoning_effort": "fast",
            "temperature": 0.2,
            "max_tokens": 256,
        }
        service.system_prompt = "你是瓦力。"
        service.model = "qwen3.6-35b-a3b"
        service.client = MagicMock()
        service.client.chat.completions.create.return_value = _FakeResponse()

        list(service.chat_stream("你好", tools_enabled=False))

        kwargs = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})
        sys.modules.pop("services.llm.llm_service", None)

    def test_zhipu_toggle_model_fast_mode_disables_thinking(self):
        fake_dispatcher = _complete_fake_dispatcher(
            types.ModuleType("services.llm.tool_dispatcher")
        )
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.llm.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm.llm_service", None)
            from services.llm.llm_service import LLMService

        service = object.__new__(LLMService)
        service.settings = {
            "provider": "zhipu",
            "reasoning_effort": "fast",
            "temperature": 0.2,
            "max_tokens": 256,
        }
        service.system_prompt = "你是瓦力。"
        service.settings["model"] = "glm-4.7"
        service.model = "glm-4.7"
        service.client = MagicMock()
        service.client.chat.completions.create.return_value = _FakeResponse()

        list(service.chat_stream("你好", tools_enabled=False))

        kwargs = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        sys.modules.pop("services.llm.llm_service", None)

    def test_zhipu_glm_45_air_fast_mode_disables_thinking(self):
        from services.llm.llm_request_options import reasoning_request_options

        options = reasoning_request_options({
            "provider": "zhipu",
            "model": "glm-4.5-air",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {"extra_body": {"thinking": {"type": "disabled"}}})

    def test_doubao_fast_mode_disables_thinking(self):
        from services.llm.llm_request_options import reasoning_request_options

        options = reasoning_request_options({
            "provider": "doubao",
            "model": "doubao-seed-2-0-lite-260215",
            "reasoning_effort": "fast",
        })

        self.assertEqual(options, {"extra_body": {"thinking": {"type": "disabled"}}})

    def test_doubao_default_mode_preserves_model_default(self):
        from services.llm.llm_request_options import reasoning_request_options

        options = reasoning_request_options({
            "provider": "doubao",
            "model": "doubao-seed-2-0-lite-260215",
            "reasoning_effort": "default",
        })

        self.assertEqual(options, {})

    def test_zhipu_fixed_thinking_model_keeps_supported_request_shape(self):
        from services.llm.llm_request_options import reasoning_request_options

        options = reasoning_request_options({
            "provider": "zhipu",
            "model": "glm-4.1v-thinking-flashx",
            "reasoning_effort": "fast",
        })
        self.assertEqual(options, {})

    def test_retry_can_raise_token_budget_without_changing_config(self):
        fake_dispatcher = _complete_fake_dispatcher(
            types.ModuleType("services.llm.tool_dispatcher")
        )
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.llm.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm.llm_service", None)
            from services.llm.llm_service import LLMService

        service = object.__new__(LLMService)
        service.settings = {"temperature": 0.2, "max_tokens": 256}
        service.system_prompt = "你是瓦力。"
        service.model = "glm-4.1v-thinking-flashx"
        service.client = MagicMock()
        service.client.chat.completions.create.return_value = _FakeResponse()

        list(service.chat_stream("重试", tools_enabled=False, max_tokens_override=512))

        kwargs = service.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertEqual(service.settings["max_tokens"], 256)
        sys.modules.pop("services.llm.llm_service", None)


class CameraIntentTests(unittest.TestCase):
    def test_inspection_phrases_are_detected(self):
        from services.vision.camera_frame import (
            is_camera_inspection_request,
            is_camera_photo_request,
        )

        self.assertTrue(is_camera_inspection_request("瓦力你看一下这是个什么东西"))
        self.assertTrue(is_camera_inspection_request("你前面有什么"))
        self.assertTrue(is_camera_inspection_request("你看我手里拿了什么？"))
        self.assertTrue(is_camera_inspection_request("看看我手里有什么"))
        self.assertTrue(is_camera_inspection_request("你从看我手里拿的是什么？"))
        self.assertTrue(is_camera_inspection_request("瞅瞅我拿的是啥"))
        self.assertFalse(is_camera_inspection_request("帮我拍张照片"))
        self.assertTrue(is_camera_photo_request("帮我拍张照片"))
        self.assertTrue(is_camera_photo_request("take a picture"))
        self.assertFalse(is_camera_inspection_request("瓦力看着我"))
        self.assertFalse(is_camera_inspection_request("跟着我"))
        self.assertFalse(is_camera_inspection_request("我想看看电影"))
        self.assertFalse(is_camera_photo_request("别拍照"))
        self.assertFalse(is_camera_photo_request("不需要帮我拍张照"))
        self.assertFalse(is_camera_inspection_request("不要看一下前面"))
        self.assertFalse(is_camera_inspection_request("看看前面就不用了"))
        self.assertFalse(is_camera_inspection_request("别看我手里的东西"))


class CameraFrameProviderTests(unittest.TestCase):
    def test_echoed_request_size_does_not_make_old_streaming_dimensions_ready(self):
        import threading
        import time
        from services.vision import camera_frame
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                provider, node, image_type = self._provider()
                result = []
                def capture():
                    result.append(provider.capture_stream(duration_ms=100, fps=15, on_frame=lambda _: None,
                                  timeout=0.2) if streaming else provider.capture(timeout=0.2))
                with patch.object(camera_frame, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.6):
                    thread = threading.Thread(target=capture)
                    thread.start()
                    until = time.monotonic() + 1
                    while not node.publisher.messages and time.monotonic() < until:
                        time.sleep(0.005)
                    node.callbacks["/camera_capture_status"](self._ack(node, state="streaming", actual_resolution=(640, 480)))
                    time.sleep(0.25)
                    self.assertTrue(thread.is_alive())
                    node.callbacks["/camera_frame"](image_type())
                    thread.join(timeout=1)
                self.assertEqual(result, [b"camera-jpeg"])

    def test_status_ack_cannot_trigger_a_lease_renewal_feedback_loop(self):
        import queue
        import threading
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                provider, node, _ = self._provider()
                commands = queue.Queue()
                stopped = threading.Event()
                original_publish = node.publisher.publish
                def publish(message):
                    original_publish(message)
                    if json.loads(message.data)["action"] != "release":
                        commands.put(message)
                def acknowledge():
                    while not stopped.is_set():
                        try:
                            commands.get(timeout=0.01)
                        except queue.Empty:
                            continue
                        node.callbacks["/camera_capture_status"](self._ack(node, state="streaming"))
                worker = threading.Thread(target=acknowledge)
                worker.start()
                try:
                    with patch.object(node.publisher, "publish", side_effect=publish):
                        result = provider.capture_stream(duration_ms=100, fps=15, on_frame=lambda _: None,
                                                         timeout=0.2) if streaming else provider.capture(timeout=0.2)
                    self.assertIsNone(result)
                    self.assertEqual([json.loads(m.data)["action"] for m in node.publisher.messages],
                                     ["acquire", "release"])
                finally:
                    stopped.set()
                    worker.join(timeout=1)

    def test_new_driver_starting_ack_uses_initialization_budget_without_reconfiguration_flag(self):
        import threading
        import time
        from services.vision import camera_frame
        provider, node, image_type = self._provider()
        result = []
        with patch.object(camera_frame, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.6):
            thread = threading.Thread(target=lambda: result.append(provider.capture(timeout=0.2)))
            thread.start()
            until = time.monotonic() + 1
            while not node.publisher.messages and time.monotonic() < until:
                time.sleep(0.005)
            node.callbacks["/camera_capture_status"](self._ack(node))
            time.sleep(0.25)
            self.assertTrue(thread.is_alive())
            node.callbacks["/camera_frame"](image_type())
            thread.join(timeout=1)
        self.assertEqual(result, [b"camera-jpeg"])

    def test_stale_streaming_then_current_reconfiguration_keeps_full_budget(self):
        import threading
        import time
        from services.vision import camera_frame
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                provider, node, image_type = self._provider()
                result = []
                emitted = []
                def capture():
                    result.append(provider.capture_stream(duration_ms=100, fps=15, on_frame=emitted.append,
                                  timeout=0.2, request_timeout=0.2) if streaming else provider.capture(timeout=0.2))
                with patch.object(camera_frame, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.6):
                    thread = threading.Thread(target=capture)
                    thread.start()
                    until = time.monotonic() + 1
                    while not node.publisher.messages and time.monotonic() < until:
                        time.sleep(0.005)
                    node.callbacks["/camera_capture_status"](type("Status", (), {
                        "data": '{"state":"streaming","reconfiguring":false}'
                    })())
                    node.callbacks["/camera_capture_status"](self._ack(node, state="streaming"))
                    time.sleep(0.02)
                    node.callbacks["/camera_capture_status"](self._ack(node, reconfiguring=True))
                    time.sleep(0.25)
                    self.assertTrue(thread.is_alive())
                    node.callbacks["/camera_frame"](image_type(b"wrong-size"))
                    time.sleep(0.02)
                    self.assertTrue(thread.is_alive())
                    node.callbacks["/camera_frame"](image_type())
                    thread.join(timeout=1)
                self.assertEqual(result, [b"camera-jpeg"])
                if streaming:
                    self.assertTrue(emitted)
                    self.assertNotIn(b"wrong-size", emitted)
                self.assertEqual(json.loads(node.publisher.messages[-1].data)["action"], "release")
                self.assertEqual(provider._requests, {})

    def test_foreign_or_wrong_resolution_status_does_not_acknowledge_request(self):
        import threading
        import time
        from services.vision import camera_frame
        for streaming in (False, True):
            for mismatch in ("client", "resolution"):
                with self.subTest(streaming=streaming, mismatch=mismatch):
                    provider, node, _ = self._provider()
                    result = []
                    def capture():
                        result.append(provider.capture_stream(duration_ms=100, fps=15, on_frame=lambda _: None,
                                      timeout=0.2) if streaming else provider.capture(timeout=0.2))
                    with patch.object(camera_frame, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.6):
                        started = time.monotonic()
                        thread = threading.Thread(target=capture)
                        thread.start()
                        until = time.monotonic() + 1
                        while not node.publisher.messages and time.monotonic() < until:
                            time.sleep(0.005)
                        status = self._ack(node, reconfiguring=True,
                                           client_id="another-request" if mismatch == "client" else None,
                                           resolution=(640, 480) if mismatch == "resolution" else None)
                        node.callbacks["/camera_capture_status"](status)
                        thread.join(timeout=0.4)
                        self.assertEqual(result, [None])
                        self.assertLess(time.monotonic() - started, 0.5)

    def test_repeated_reconfiguration_ack_does_not_reset_total_wait(self):
        import threading
        import time
        from services.vision import camera_frame
        provider, node, _ = self._provider()
        result = []
        with patch.object(camera_frame, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.6):
            thread = threading.Thread(target=lambda: result.append(provider.capture(timeout=0.2)))
            thread.start()
            until = time.monotonic() + 1
            while not node.publisher.messages and time.monotonic() < until:
                time.sleep(0.005)
            status = self._ack(node, reconfiguring=True)
            node.callbacks["/camera_capture_status"](status)
            for _ in range(4):
                time.sleep(0.1)
                node.callbacks["/camera_capture_status"](status)
            thread.join(timeout=0.4)
            self.assertEqual(result, [None])

    def _ack(self, node, *, state="starting", reconfiguring=False, client_id=None, resolution=None, actual_resolution=None):
        import json
        command = json.loads(node.publisher.messages[0].data)
        width, height = resolution or (command.get("width", 640), command.get("height", 480))
        return type("Status", (), {"data": json.dumps({
            "state": state, "reconfiguring": reconfiguring,
            "width": (actual_resolution or (width, height))[0],
            "height": (actual_resolution or (width, height))[1],
            "request": {"client_id": client_id or command["client_id"], "width": width, "height": height},
        })})()

    def test_resolution_change_waits_for_driver_startup_instead_of_short_frame_timeout(self):
        provider, node, image_type = self._provider()
        import sys
        import threading
        import time
        from unittest.mock import patch
        module = sys.modules[provider.__class__.__module__]
        result = []
        with patch.object(module, "CAMERA_FIRST_FRAME_TIMEOUT_SEC", 0.5, create=True):
            thread = threading.Thread(target=lambda: result.append(provider.capture(timeout=0.05)))
            thread.start()
            time.sleep(0.01)
            node.callbacks["/camera_capture_status"](self._ack(node, reconfiguring=True))
            time.sleep(0.25)
            self.assertTrue(thread.is_alive())
            node.callbacks["/camera_frame"](image_type())
            thread.join(timeout=1)
        self.assertEqual(result, [b"camera-jpeg"])

    def test_photo_is_rejected_during_tracking_and_does_not_acquire_high_resolution(self):
        provider, node, _image_type = self._provider()
        provider._status_tracking = True
        with self.assertRaisesRegex(RuntimeError, "退出"):
            provider.capture(timeout=0.2)
        commands = [__import__("json").loads(message.data) for message in node.publisher.messages]
        self.assertEqual([command["action"] for command in commands], ["release"])

    def _provider(self):
        from services.vision import camera_frame

        class FakeCompressedImage:
            encoding = "jpeg"

            def __init__(self, data=b"camera-jpeg"):
                self.data = data

        class FakeString:
            def __init__(self, data=""):
                self.data = data

        class FakePublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        class FakeNode:
            def __init__(self):
                self.publisher = FakePublisher()
                self.publisher_topic = ""
                self.subscription_topics = []
                self.subscription_types = {}
                self.callbacks = {}

            def create_publisher(self, _message_type, topic, _qos):
                self.publisher_topic = topic
                return self.publisher

            def create_subscription(self, message_type, topic, callback, _qos):
                self.subscription_topics.append(topic)
                self.subscription_types[topic] = message_type
                self.callbacks[topic] = callback
                return object()

        node = FakeNode()
        image_patch = patch.object(camera_frame, "CompressedImage", FakeCompressedImage)
        string_patch = patch.object(camera_frame, "String", FakeString)
        jpeg_patch = patch.object(
            camera_frame,
            "jpeg_from_ros_image",
            side_effect=lambda message, quality=85, validate_decode=True: (
                None if message.data == b"broken-jpeg" else bytes(message.data)
            ),
        )
        image_patch.start()
        string_patch.start()
        jpeg_patch.start()
        def open_image(stream):
            context = MagicMock()
            context.__enter__.return_value.size = (640, 480) if stream.getvalue() == b"wrong-size" else (1280, 720)
            return context

        dimensions_patch = patch.object(camera_frame.Image, "open", side_effect=open_image)
        dimensions_patch.start()
        self.addCleanup(image_patch.stop)
        self.addCleanup(string_patch.stop)
        self.addCleanup(jpeg_patch.stop)
        self.addCleanup(dimensions_patch.stop)
        return (
            camera_frame.CameraFrameProvider(
                node,
                warmup_seconds=0.0,
                warmup_frames=1,
            ),
            node,
            FakeCompressedImage,
        )

    def test_provider_uses_only_camera_frame_and_capture_command_topics(self):
        provider, node, _image_type = self._provider()

        self.assertIsNotNone(provider)
        self.assertEqual(node.publisher_topic, "/camera_capture_cmd")
        self.assertEqual(
            node.subscription_topics,
            ["/camera_frame", "/camera_capture_status"],
        )
        self.assertEqual(
            node.subscription_types["/camera_frame"].__name__,
            "FakeCompressedImage",
        )

    def test_capture_acquires_waits_for_fresh_frame_and_releases(self):
        provider, node, image_type = self._provider()
        result = []

        thread = __import__("threading").Thread(
            target=lambda: result.append(provider.capture(timeout=0.8))
        )
        thread.start()
        deadline = __import__("time").monotonic() + 0.5
        while not node.publisher.messages and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.01)
        node.callbacks["/camera_frame"](image_type())
        thread.join(timeout=1.0)

        commands = [
            __import__("json").loads(message.data)
            for message in node.publisher.messages
        ]
        self.assertEqual(result, [b"camera-jpeg"])
        self.assertEqual(commands[0]["action"], "acquire")
        self.assertEqual((commands[0]["width"], commands[0]["height"]), (1280, 720))
        self.assertEqual(commands[-1]["action"], "release")

    def test_capture_timeout_releases_without_uvc_fallback(self):
        provider, node, _image_type = self._provider()
        self.assertIsNone(provider.capture(timeout=0.2))

        commands = [
            __import__("json").loads(message.data)
            for message in node.publisher.messages
        ]
        self.assertEqual(commands[-1]["action"], "release")
        self.assertFalse(hasattr(provider, "_capture_uvc"))

    def test_capture_ignores_invalid_jpeg_before_returning_a_valid_frame(self):
        provider, node, image_type = self._provider()
        result = []

        thread = __import__("threading").Thread(
            target=lambda: result.append(provider.capture(timeout=0.8))
        )
        thread.start()
        deadline = __import__("time").monotonic() + 0.5
        while not node.publisher.messages and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.01)
        node.callbacks["/camera_frame"](image_type(b"broken-jpeg"))
        __import__("time").sleep(0.03)
        self.assertTrue(thread.is_alive())
        node.callbacks["/camera_frame"](image_type(b"valid-jpeg"))
        thread.join(timeout=1.0)

        self.assertEqual(result, [b"valid-jpeg"])

    def test_capture_waits_for_camera_warmup_frames(self):
        provider, node, image_type = self._provider()
        provider._warmup_seconds = 0.04
        provider._warmup_frames = 3
        result = []

        thread = __import__("threading").Thread(
            target=lambda: result.append(provider.capture(timeout=0.8))
        )
        thread.start()
        deadline = __import__("time").monotonic() + 0.5
        while not node.publisher.messages and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.01)
        node.callbacks["/camera_frame"](image_type(b"frame-1"))
        __import__("time").sleep(0.02)
        node.callbacks["/camera_frame"](image_type(b"frame-2"))
        __import__("time").sleep(0.03)
        self.assertTrue(thread.is_alive())
        node.callbacks["/camera_frame"](image_type(b"frame-3"))
        thread.join(timeout=1.0)

        self.assertEqual(result, [b"frame-3"])

    def test_capture_stream_reuses_one_lease_and_returns_last_complete_frame(self):
        provider, node, image_type = self._provider()
        emitted = []
        result = []
        thread = __import__("threading").Thread(
            target=lambda: result.append(provider.capture_stream(
                duration_ms=120,
                fps=20,
                on_frame=emitted.append,
                timeout=0.5,
                request_timeout=0.5,
            ))
        )
        thread.start()
        deadline = __import__("time").monotonic() + 0.5
        while not node.publisher.messages and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.01)
        node.callbacks["/camera_frame"](image_type(b"frame-1"))
        __import__("time").sleep(0.04)
        node.callbacks["/camera_frame"](image_type(b"frame-2"))
        __import__("time").sleep(0.05)
        node.callbacks["/camera_frame"](image_type(b"frame-3"))
        thread.join(timeout=1.0)

        commands = [
            __import__("json").loads(message.data)
            for message in node.publisher.messages
        ]
        self.assertEqual(result, [b"frame-3"])
        self.assertGreaterEqual(len(emitted), 2)
        self.assertEqual(commands[0]["action"], "acquire")
        self.assertEqual(commands[-1]["action"], "release")

    def test_manager_ack_starts_a_fresh_frame_timeout(self):
        provider, node, image_type = self._provider()
        result = []
        started_at = __import__("time").monotonic()

        thread = __import__("threading").Thread(
            target=lambda: result.append(
                provider.capture(timeout=0.25, request_timeout=0.4)
            )
        )
        thread.start()
        __import__("time").sleep(0.2)
        status_message = self._ack(node, state="streaming")
        node.callbacks["/camera_capture_status"](status_message)
        __import__("time").sleep(0.15)
        node.callbacks["/camera_frame"](image_type())
        thread.join(timeout=1.0)

        self.assertGreater(__import__("time").monotonic() - started_at, 0.25)
        self.assertEqual(result, [b"camera-jpeg"])

    def test_repeated_manager_status_does_not_extend_frame_timeout(self):
        provider, node, _image_type = self._provider()
        result = []
        started_at = __import__("time").monotonic()

        thread = __import__("threading").Thread(
            target=lambda: result.append(
                provider.capture(timeout=0.2, request_timeout=0.4)
            )
        )
        thread.start()
        __import__("time").sleep(0.03)
        status_message = self._ack(node, state="streaming")
        node.callbacks["/camera_capture_status"](status_message)
        __import__("time").sleep(0.1)
        node.callbacks["/camera_capture_status"](status_message)
        thread.join(timeout=0.3)

        elapsed = __import__("time").monotonic() - started_at
        self.assertEqual(result, [None])
        self.assertLess(elapsed, 0.32)


if __name__ == "__main__":
    unittest.main()
