import base64
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeResponse:
    def __iter__(self):
        delta = types.SimpleNamespace(content="看起来是一只杯子。", tool_calls=None)
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class LLMServiceVisionTests(unittest.TestCase):
    def test_image_is_sent_as_openai_vision_content_without_tools(self):
        fake_dispatcher = types.ModuleType("services.tool_dispatcher")
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm_service", None)
            from services.llm_service import LLMService

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
        result = list(service.chat_stream("这是什么？", image_base64=payload, tools_enabled=False))

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
        sys.modules.pop("services.llm_service", None)

    def test_aliyun_fast_mode_disables_thinking(self):
        fake_dispatcher = types.ModuleType("services.tool_dispatcher")
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm_service", None)
            from services.llm_service import LLMService

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
        sys.modules.pop("services.llm_service", None)

    def test_zhipu_toggle_model_fast_mode_disables_thinking(self):
        fake_dispatcher = types.ModuleType("services.tool_dispatcher")
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm_service", None)
            from services.llm_service import LLMService

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
        sys.modules.pop("services.llm_service", None)

    def test_zhipu_fixed_thinking_model_keeps_supported_request_shape(self):
        from services.llm_request_options import reasoning_request_options

        options = reasoning_request_options({
            "provider": "zhipu",
            "model": "glm-4.1v-thinking-flashx",
            "reasoning_effort": "fast",
        })
        self.assertEqual(options, {})

    def test_retry_can_raise_token_budget_without_changing_config(self):
        fake_dispatcher = types.ModuleType("services.tool_dispatcher")
        fake_dispatcher.get_tools = lambda: []

        class FakeAccumulator:
            def feed(self, _delta):
                pass

            def flush(self):
                return []

        fake_dispatcher.ToolCallAccumulator = FakeAccumulator
        with patch.dict(sys.modules, {"services.tool_dispatcher": fake_dispatcher}):
            sys.modules.pop("services.llm_service", None)
            from services.llm_service import LLMService

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
        sys.modules.pop("services.llm_service", None)


class CameraIntentTests(unittest.TestCase):
    def test_inspection_phrases_are_detected(self):
        from services.camera_frame import is_camera_inspection_request

        self.assertTrue(is_camera_inspection_request("瓦力你看一下这是个什么东西"))
        self.assertTrue(is_camera_inspection_request("你前面有什么"))
        self.assertFalse(is_camera_inspection_request("瓦力看着我"))
        self.assertFalse(is_camera_inspection_request("跟着我"))


class CameraFrameProviderTests(unittest.TestCase):
    def test_cached_frame_is_returned(self):
        from services.camera_frame import CameraFrameProvider

        node = MagicMock()
        provider = CameraFrameProvider(node)
        provider._frames["/image"] = (b"cached-jpeg", __import__("time").monotonic())
        self.assertEqual(provider.capture(timeout=0.01), b"cached-jpeg")

    def test_corrected_topic_has_priority_over_raw_image(self):
        from services.camera_frame import CameraFrameProvider

        node = MagicMock()
        provider = CameraFrameProvider(node)
        now = __import__("time").monotonic()
        provider._frames["/image"] = (b"raw", now)
        provider._frames["/image_padded_jpeg"] = (b"corrected", now)
        self.assertEqual(provider.capture(timeout=0.01), b"corrected")

    def test_uvc_fallback_is_used_when_cache_is_empty(self):
        from services.camera_frame import CameraFrameProvider

        node = MagicMock()
        provider = CameraFrameProvider(node)
        with patch.object(provider, "_capture_uvc", return_value=b"uvc-jpeg"):
            self.assertEqual(provider.capture(timeout=0.01), b"uvc-jpeg")


if __name__ == "__main__":
    unittest.main()
