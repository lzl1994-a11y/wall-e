from services.llm_response_policy import LLMResponsePolicy


def test_policy_identifies_long_form_and_uses_configured_token_floor():
    assert LLMResponsePolicy.is_long_form_request("背一下琵琶行")
    assert LLMResponsePolicy.max_tokens(4096, long_form=True) == 4096
    assert LLMResponsePolicy.max_tokens("bad", long_form=True) == 2048
    assert LLMResponsePolicy.max_tokens(4096, long_form=False) is None


def test_policy_strips_correction_metadata_without_suppressing_ordinary_speech():
    assert LLMResponsePolicy.sanitize_speech_text(
        "修正文本\n你好，瓦力。\n很高兴见到你。"
    ) == "很高兴见到你。"
    assert LLMResponsePolicy.sanitize_speech_text("修正文本是什么意思？") == "修正文本是什么意思？"
    assert LLMResponsePolicy.sanitize_speech_text("[corrected_text] 你好") == ""
