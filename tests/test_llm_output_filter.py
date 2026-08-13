import unittest

from services.llm_output_filter import VisibleAnswerFilter


class VisibleAnswerFilterTests(unittest.TestCase):
    def collect(self, chunks):
        output_filter = VisibleAnswerFilter()
        visible = [output_filter.feed(chunk) for chunk in chunks]
        visible.append(output_filter.flush())
        return "".join(visible)

    def test_plain_stream_passes_through(self):
        self.assertEqual(
            self.collect(["\u3010\u4fee\u6b63\u6587\u672c\u3011: \u4f60\u597d\n", "\u53c8\u6765\u70e6\u6211\uff1f"]),
            "\u3010\u4fee\u6b63\u6587\u672c\u3011: \u4f60\u597d\n\u53c8\u6765\u70e6\u6211\uff1f",
        )

    def test_think_block_is_removed_and_answer_is_kept(self):
        self.assertEqual(
            self.collect([
                "<think>\u8fd9\u662f\u5206\u6790\u3002</think>",
                "<answer>\u3010\u4fee\u6b63\u6587\u672c\u3011: \u4f60\u597d\n\u53c8\u6765\u70e6\u6211\uff1f</answer>",
            ]),
            "\u3010\u4fee\u6b63\u6587\u672c\u3011: \u4f60\u597d\n\u53c8\u6765\u70e6\u6211\uff1f",
        )

    def test_tags_split_across_chunks_are_filtered(self):
        self.assertEqual(
            self.collect([
                "<thi",
                "nk>\u5206\u6790\u5185\u5bb9</thi",
                "nk>\n<ans",
                "wer>\u7b2c\u4e00\u884c\uff1a\u4fee\u6b63\u6587\u672c \u4f60\u597d\n",
                "\u7b2c\u4e8c\u884c\uff1a\u6211\u597d\u5f97\u5f88\u3002</ans",
                "wer>",
            ]),
            "\u7b2c\u4e00\u884c\uff1a\u4fee\u6b63\u6587\u672c \u4f60\u597d\n\u7b2c\u4e8c\u884c\uff1a\u6211\u597d\u5f97\u5f88\u3002",
        )

    def test_untagged_answer_after_think_is_kept(self):
        self.assertEqual(
            self.collect(["<think>\u5206\u6790</think>", "\n\u6700\u7ec8\u56de\u590d\u3002"]),
            "\u6700\u7ec8\u56de\u590d\u3002",
        )


if __name__ == "__main__":
    unittest.main()
