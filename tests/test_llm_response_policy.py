from services.llm.llm_response_policy import LLMResponsePolicy


def test_policy_identifies_long_form_and_uses_configured_token_floor():
    assert LLMResponsePolicy.is_long_form_request("背一下琵琶行")
    assert LLMResponsePolicy.is_long_form_request("念一遍整首诗歌")
    assert LLMResponsePolicy.is_long_form_request("朗读全文给我听")
    assert not LLMResponsePolicy.is_long_form_request("今天天气怎么样")
    assert not LLMResponsePolicy.is_long_form_request("向前走")

    assert LLMResponsePolicy.max_tokens(4096, long_form=True) == 4096
    assert LLMResponsePolicy.max_tokens(1024, long_form=True) == 2048
    assert LLMResponsePolicy.max_tokens("bad", long_form=True) == 2048
    assert LLMResponsePolicy.max_tokens(4096, long_form=False) is None


def test_extract_corrected_text():
    assert (
        LLMResponsePolicy.extract_corrected_text("【修正文本】背一下琵琶行。")
        == "背一下琵琶行。"
    )
    assert (
        LLMResponsePolicy.extract_corrected_text("[corrected_text] hello world")
        == "hello world"
    )
    assert (
        LLMResponsePolicy.extract_corrected_text("修正文本: 瓦力向前走")
        == "瓦力向前走"
    )
    assert (
        LLMResponsePolicy.extract_corrected_text("第一行：【校正文本】你好")
        == "你好"
    )
    assert (
        LLMResponsePolicy.extract_corrected_text("> 【纠错文本】“今天好吗”")
        == "今天好吗"
    )
    # 纯标签无正文返回 None
    assert LLMResponsePolicy.extract_corrected_text("【修正文本】") is None
    assert LLMResponsePolicy.extract_corrected_text("修正文本:") is None
    # 普通非纠错语句返回 None
    assert LLMResponsePolicy.extract_corrected_text("修正文本是什么意思？") is None
    assert LLMResponsePolicy.extract_corrected_text("今天天气真好") is None
    assert LLMResponsePolicy.extract_corrected_text("") is None


def test_is_correction_label_only():
    assert LLMResponsePolicy.is_correction_label_only("修正文本")
    assert LLMResponsePolicy.is_correction_label_only("【纠错文本】")
    assert LLMResponsePolicy.is_correction_label_only("[corrected_text]")
    assert LLMResponsePolicy.is_correction_label_only("> 识别修正:")
    assert not LLMResponsePolicy.is_correction_label_only("修正文本 背一下琵琶行")
    assert not LLMResponsePolicy.is_correction_label_only("今天天气真好")
    assert not LLMResponsePolicy.is_correction_label_only("")


def test_strip_answer_prefix():
    assert LLMResponsePolicy.strip_answer_prefix("回答：你好呀") == "你好呀"
    assert LLMResponsePolicy.strip_answer_prefix("回复: 收到") == "收到"
    assert LLMResponsePolicy.strip_answer_prefix("最终回答：完成") == "完成"
    assert LLMResponsePolicy.strip_answer_prefix("第二行: 这是第二行") == "这是第二行"
    assert LLMResponsePolicy.strip_answer_prefix("你好呀") == "你好呀"


def test_strip_correction_line():
    # 两行格式
    assert (
        LLMResponsePolicy.strip_correction_line("【修正文本】背一下琵琶行\n浔阳江头夜送客。")
        == "浔阳江头夜送客。"
    )
    # 三行格式
    assert (
        LLMResponsePolicy.strip_correction_line(
            "修正文本\n背一下琵琶行。\n回答：浔阳江头夜送客。"
        )
        == "浔阳江头夜送客。"
    )
    # 单行无换行
    assert LLMResponsePolicy.strip_correction_line("今天天气很好") == "今天天气很好"
    # 普通多行文本不剥离
    assert (
        LLMResponsePolicy.strip_correction_line("第一行正文\n第二行正文")
        == "第一行正文\n第二行正文"
    )


def test_clean_tts_text():
    assert (
        LLMResponsePolicy.clean_tts_text("你好，瓦力！今天好吗？")
        == "你好，瓦力！今天好吗？"
    )
    # 清洗掉非语音字符（如 markdown、特殊符号），保留中文全角括号及中文标点
    assert (
        LLMResponsePolicy.clean_tts_text("**瓦力**#1 <online> （测试）")
        == "瓦力1 online （测试）"
    )
    # 半角括号被清洗
    assert (
        LLMResponsePolicy.clean_tts_text("(测试) [hello] {world}")
        == "测试 hello world"
    )


def test_sanitize_speech_text():
    assert (
        LLMResponsePolicy.sanitize_speech_text(
            "修正文本\n你好，瓦力。\n很高兴见到你。"
        )
        == "很高兴见到你。"
    )
    assert (
        LLMResponsePolicy.sanitize_speech_text("回答：今天天气很晴朗。")
        == "今天天气很晴朗。"
    )
    assert (
        LLMResponsePolicy.sanitize_speech_text(
            "【修正文本】背一下琵琶行\n回答：浔阳江头夜送客。"
        )
        == "浔阳江头夜送客。"
    )
    # 单行仅含纠错标签及文本时，不触发 TTS 播报
    assert LLMResponsePolicy.sanitize_speech_text("[corrected_text] 你好") == ""
    assert LLMResponsePolicy.sanitize_speech_text("【修正文本】背一下琵琶行") == ""
    # 用户询问修正文本概念，正常保留
    assert (
        LLMResponsePolicy.sanitize_speech_text("修正文本是什么意思？")
        == "修正文本是什么意思？"
    )


def test_clean_visual_answer():
    assert (
        LLMResponsePolicy.clean_visual_answer("```\n这是一只小猫咪。\n```")
        == "这是一只小猫咪。"
    )
    assert (
        LLMResponsePolicy.clean_visual_answer("【修正文本】看到前方有一本书。")
        == "看到前方有一本书。"
    )
    assert (
        LLMResponsePolicy.clean_visual_answer("ai: 画面中有一个杯子")
        == "画面中有一个杯子"
    )
    assert (
        LLMResponsePolicy.clean_visual_answer("AI：检测到红色方块")
        == "检测到红色方块"
    )
    assert (
        LLMResponsePolicy.clean_visual_answer("修正文本: 门是关着的。")
        == "门是关着的。"
    )
