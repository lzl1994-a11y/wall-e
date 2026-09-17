"""Unit tests for voice request routing and preparation service."""

import unittest
from unittest.mock import patch

from services.llm_request_preparation import (
    ROUTE_CAMERA_INSPECTION,
    ROUTE_CAMERA_PHOTO,
    ROUTE_CONDITIONAL_TASK,
    ROUTE_ORDINARY_DIALOG,
    ROUTE_SAFETY_ACTION,
    LLMRequestPreparation,
    PreparedVoiceRequest,
    prepare_voice_request,
)


class LLMRequestPreparationTests(unittest.TestCase):
    def test_deterministic_safety_action_priority(self):
        prompts = [
            ("停止跟随", "set_tracking_mode", {"mode": "idle"}),
            ("不要再看着我了", "set_tracking_mode", {"mode": "idle"}),
            ("别再盯着我了", "set_tracking_mode", {"mode": "idle"}),
        ]
        for prompt, expected_name, expected_args in prompts:
            decision = prepare_voice_request(prompt)
            self.assertEqual(decision.route, ROUTE_SAFETY_ACTION)
            self.assertTrue(decision.is_safety_action)
            self.assertEqual(decision.safety_action_name, expected_name)
            self.assertEqual(decision.safety_action_arguments, expected_args)
            self.assertIsNone(decision.augmented_prompt)
            self.assertIsNone(decision.max_tokens_override)
            self.assertTrue(decision.tools_enabled)

    def test_conditional_task_priority_over_camera_photo_and_inspection(self):
        compound_prompts = [
            "如果看到红色物体，就拍张照",
            "只要检测到人，就帮我拍一张",
            "如果看见杯子，就识别一下",
            "看看前面，如果有人挥手你就点头",
            "当发现障碍物时，就拍张照",
        ]
        for prompt in compound_prompts:
            decision = prepare_voice_request(prompt)
            self.assertEqual(
                decision.route,
                ROUTE_CONDITIONAL_TASK,
                f"Prompt '{prompt}' should route to conditional_task instead of photo/inspection",
            )
            self.assertTrue(decision.is_conditional_task)
            self.assertFalse(decision.is_camera_photo)
            self.assertFalse(decision.is_camera_inspection)
            self.assertIsNone(decision.augmented_prompt)
            self.assertIsNone(decision.max_tokens_override)

    def test_standalone_camera_photo(self):
        photo_prompts = [
            "拍照",
            "拍张照",
            "拍一张",
            "帮我拍张照",
            "take a photo",
        ]
        for prompt in photo_prompts:
            decision = prepare_voice_request(prompt)
            self.assertEqual(decision.route, ROUTE_CAMERA_PHOTO)
            self.assertTrue(decision.is_camera_photo)
            self.assertFalse(decision.is_conditional_task)
            self.assertFalse(decision.is_ordinary_dialog)
            self.assertIsNone(decision.safety_action_name)
            self.assertIsNone(decision.augmented_prompt)
            self.assertIsNone(decision.max_tokens_override)

    def test_standalone_camera_inspection(self):
        inspection_prompts = [
            "帮我看看前面是什么",
            "识别一下这个是什么",
            "看一下前面有人吗",
        ]
        for prompt in inspection_prompts:
            decision = prepare_voice_request(prompt)
            self.assertEqual(decision.route, ROUTE_CAMERA_INSPECTION)
            self.assertTrue(decision.is_camera_inspection)
            self.assertFalse(decision.is_conditional_task)
            self.assertFalse(decision.is_camera_photo)
            self.assertIsNone(decision.safety_action_name)
            self.assertIsNone(decision.augmented_prompt)
            self.assertIsNone(decision.max_tokens_override)

    def test_ordinary_short_dialog_prompt_and_token_decision(self):
        prompt = "今天天气怎么样"
        decision = prepare_voice_request(prompt, configured_tokens=512)
        self.assertEqual(decision.route, ROUTE_ORDINARY_DIALOG)
        self.assertTrue(decision.is_ordinary_dialog)
        self.assertFalse(decision.is_long_form)
        self.assertTrue(decision.tools_enabled)
        self.assertIsNone(decision.max_tokens_override)
        self.assertIsNotNone(decision.augmented_prompt)

        augmented = decision.augmented_prompt
        self.assertTrue(augmented.startswith(f"原始 ASR 文本：{prompt}\n"))
        self.assertIn("拼音参考：jin tian tian qi zen me yang", augmented)
        self.assertIn(
            "请结合对话上下文和拼音静默理解用户本意，然后直接回答。",
            augmented,
        )
        self.assertIn("普通对话保持一到两句、简短自然。", augmented)
        self.assertNotIn("这是朗读、背诵或完整内容请求", augmented)

    def test_long_form_request_min_2048_tokens(self):
        long_prompts = [
            "背诵一下将进酒",
            "朗读全文",
            "念一遍从头到尾",
        ]
        for prompt in long_prompts:
            # When model config is 0 or less than 2048, override must be at least 2048
            decision_zero = prepare_voice_request(prompt, configured_tokens=0)
            self.assertEqual(decision_zero.route, ROUTE_ORDINARY_DIALOG)
            self.assertTrue(decision_zero.is_long_form)
            self.assertEqual(decision_zero.max_tokens_override, 2048)

            decision_small = prepare_voice_request(prompt, configured_tokens=512)
            self.assertEqual(decision_small.max_tokens_override, 2048)

            self.assertIn(
                "这是朗读、背诵或完整内容请求。请连续完整输出用户要求的正文，"
                "不要只给标题、简介或开头一句；除非用户明确只要片段。",
                decision_small.augmented_prompt,
            )
            self.assertNotIn("普通对话保持一到两句、简短自然。", decision_small.augmented_prompt)

    def test_model_config_larger_retains_larger_max_tokens(self):
        prompt = "背诵一下长恨歌全文"
        decision_4096 = prepare_voice_request(prompt, configured_tokens=4096)
        self.assertEqual(decision_4096.max_tokens_override, 4096)

        decision_settings = prepare_voice_request(
            prompt,
            model_settings={"max_tokens": 8192},
        )
        self.assertEqual(decision_settings.max_tokens_override, 8192)

    def test_non_integer_config_safe_fallback(self):
        prompt = "背一遍蜀道难"
        # Non-integer strings or None or objects should safely fall back to 2048 for long form
        for invalid_tokens in ["invalid_token_count", None, 3.14, [], {}]:
            decision = prepare_voice_request(prompt, configured_tokens=invalid_tokens)
            self.assertEqual(
                decision.max_tokens_override,
                2048,
                f"Expected 2048 token fallback for {invalid_tokens!r}",
            )

        # For ordinary short dialog, max_tokens_override is always None regardless of non-int config
        decision_short = prepare_voice_request("你好啊", configured_tokens="bad_config")
        self.assertIsNone(decision_short.max_tokens_override)

    def test_service_has_zero_external_side_effects(self):
        # Verify that the decision object is frozen (immutable)
        decision = prepare_voice_request("你好")
        with self.assertRaises(Exception):
            decision.route = "something_else"

        # Verify no external calls or mutations
        sample_prompts = [
            "停止跟随",
            "如果看到红灯就停下",
            "拍照",
            "看看前面是什么",
            "讲个笑话",
        ]
        for p in sample_prompts:
            d = prepare_voice_request(p)
            self.assertIsInstance(d, PreparedVoiceRequest)
            self.assertEqual(d.user_prompt, p)

    def test_empty_or_whitespace_prompt(self):
        for empty_prompt in ["", "   ", None]:
            decision = prepare_voice_request(empty_prompt)
            self.assertEqual(decision.route, ROUTE_ORDINARY_DIALOG)
            self.assertFalse(decision.is_long_form)
            self.assertIsNone(decision.max_tokens_override)
            self.assertIsNotNone(decision.augmented_prompt)


if __name__ == "__main__":
    unittest.main()
