"""Unit tests for pure LLM photo capture decisions."""

from dataclasses import FrozenInstanceError
import inspect
import types
import unittest

from services.llm_photo_capture import (
    PhotoCaptureDecision,
    decide_photo_preview,
    photo_save_failed,
    photo_save_succeeded,
)


class LLMPhotoCaptureTests(unittest.TestCase):
    def test_busy_true_returns_camera_preview_busy(self):
        """1. busy=True 返回 camera_preview_busy。"""
        preview = types.SimpleNamespace(busy=True, last_frame=b"frame", error="")
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.error, "camera_preview_busy")
        self.assertFalse(decision.should_save)
        self.assertIsNone(decision.frame)
        self.assertTrue(decision.is_complete)
        self.assertFalse(decision.is_ready_to_save)

    def test_busy_branch_exact_wording(self):
        """2. busy 分支使用精确文案。"""
        preview = types.SimpleNamespace(busy=True, last_frame=b"frame", error="")
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.answer, "我正在拍上一张，等一下再试。")

    def test_busy_branch_does_not_access_last_frame_or_error(self):
        """3. busy 分支不访问 last_frame 与 error，只定义 busy 属性的对象验证通过。"""
        class OnlyBusyObject:
            busy = True

        decision = decide_photo_preview(OnlyBusyObject())
        self.assertEqual(decision.answer, "我正在拍上一张，等一下再试。")
        self.assertEqual(decision.error, "camera_preview_busy")
        self.assertFalse(decision.should_save)

    def test_busy_false_and_last_frame_none_returns_no_frame_wording(self):
        """4. busy=False 且 last_frame=None 时返回无画面文案。"""
        preview = types.SimpleNamespace(busy=False, last_frame=None, error="")
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.answer, "这次没有拍到，检查一下摄像头连接。")
        self.assertFalse(decision.should_save)
        self.assertIsNone(decision.frame)

    def test_no_frame_with_error_preserves_error(self):
        """5. 无画面且 preview.error 有值时原样保留。"""
        preview = types.SimpleNamespace(busy=False, last_frame=None, error="device_timeout")
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.error, "device_timeout")

    def test_no_frame_empty_error_falls_back_to_unavailable(self):
        """6. 无画面且 preview.error 为空字符串时回退 camera_frame_unavailable。"""
        preview = types.SimpleNamespace(busy=False, last_frame=None, error="")
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.error, "camera_frame_unavailable")

    def test_no_frame_none_error_falls_back_to_unavailable(self):
        """7. 无画面且 preview.error=None 时回退 camera_frame_unavailable。"""
        preview = types.SimpleNamespace(busy=False, last_frame=None, error=None)
        decision = decide_photo_preview(preview)
        self.assertEqual(decision.error, "camera_frame_unavailable")

    def test_has_last_frame_should_save_true(self):
        """8. 有 last_frame 时 should_save=True。"""
        raw_frame = b"\xff\xd8\xff\xe0...test-jpeg"
        preview = types.SimpleNamespace(busy=False, last_frame=raw_frame, error="")
        decision = decide_photo_preview(preview)
        self.assertTrue(decision.should_save)
        self.assertTrue(decision.is_ready_to_save)
        self.assertFalse(decision.is_complete)

    def test_ready_to_save_preserves_identical_frame_bytes(self):
        """9. 待保存决策保留同一个 frame bytes。"""
        raw_frame = b"binary-jpeg-data"
        preview = types.SimpleNamespace(busy=False, last_frame=raw_frame, error="")
        decision = decide_photo_preview(preview)
        self.assertIs(decision.frame, raw_frame)

    def test_ready_to_save_answer_and_error_are_none(self):
        """10. 待保存决策 answer=None、error=None。"""
        preview = types.SimpleNamespace(busy=False, last_frame=b"frame", error="old_error")
        decision = decide_photo_preview(preview)
        self.assertIsNone(decision.answer)
        self.assertIsNone(decision.error)

    def test_save_succeeded_exact_wording(self):
        """11. 保存成功决策使用精确文案。"""
        decision = photo_save_succeeded()
        self.assertEqual(decision.answer, "拍好了，照片已经保存。")
        self.assertFalse(decision.should_save)
        self.assertIsNone(decision.frame)
        self.assertTrue(decision.is_complete)

    def test_save_succeeded_error_is_none(self):
        """12. 保存成功 error=None。"""
        decision = photo_save_succeeded()
        self.assertIsNone(decision.error)

    def test_save_failed_exact_wording(self):
        """13. 保存失败决策使用精确文案。"""
        decision = photo_save_failed("disk_full")
        self.assertEqual(decision.answer, "画面拍到了，但照片保存失败了。")
        self.assertFalse(decision.should_save)
        self.assertIsNone(decision.frame)
        self.assertTrue(decision.is_complete)

    def test_save_failed_preserves_raw_error_string(self):
        """14. 保存失败保留原始错误字符串。"""
        err_msg = "[Errno 28] No space left on device"
        decision = photo_save_failed(err_msg)
        self.assertEqual(decision.error, err_msg)

    def test_decision_is_frozen_dataclass(self):
        """15. PhotoCaptureDecision 为 frozen dataclass。"""
        decision = photo_save_succeeded()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            decision.answer = "new answer"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            decision.error = "new error"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            decision.frame = b"new frame"  # type: ignore[misc]
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            decision.should_save = True  # type: ignore[misc]

    def test_missing_busy_attribute_raises_attribute_error(self):
        """16. 非法 preview 缺少 busy 属性时保持 AttributeError，不静默吞掉。"""
        class InvalidPreview:
            last_frame = b"frame"

        with self.assertRaises(AttributeError):
            decide_photo_preview(InvalidPreview())

        with self.assertRaises(AttributeError):
            decide_photo_preview(object())

    def test_service_is_pure_and_does_not_touch_ros_or_fs(self):
        """17. Service 不导入 rclpy、不访问文件系统、不调用 save_camera_photo、不发布消息。"""
        import services.llm_photo_capture as mod

        self.assertNotIn("rclpy", mod.__dict__)
        source = inspect.getsource(mod)
        self.assertNotIn("import rclpy", source)
        self.assertNotIn("from rclpy", source)
        self.assertNotIn("save_camera_photo", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("publish(", source)


if __name__ == "__main__":
    unittest.main()
