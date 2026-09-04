"""Provider-specific LLM request options shared by both voice pipelines."""

from urllib.parse import urlparse


REASONING_MODES = {"fast", "default"}
DOUBAO_PROVIDERS = {"doubao", "volcengine", "ark"}
MIMO_PROVIDERS = {"mimo", "xiaomi", "xiaomi_mimo"}


def _is_mimo_endpoint(settings):
    provider = str(settings.get("provider", "")).strip().lower()
    model = str(settings.get("model", "")).strip().lower()
    # MiMo ASR/TTS share the same API hostname but do not accept the chat
    # model's thinking switch.
    if model.startswith("mimo-") and (
        model.endswith("-asr") or "-tts" in model
    ):
        return False
    if provider in MIMO_PROVIDERS or model.startswith("mimo-"):
        return True
    try:
        hostname = (urlparse(str(settings.get("url", ""))).hostname or "").lower()
    except ValueError:
        return False
    return hostname == "xiaomimimo.com" or hostname.endswith(".xiaomimimo.com")


def normalize_tool_choice(settings, tool_choice):
    """Map tool selection to values the configured provider documents."""
    if _is_mimo_endpoint(settings) and tool_choice != "auto":
        # MiMo Chat currently documents only ``auto``.  Its backend silently
        # drops other values today, but normalizing here avoids depending on
        # that compatibility behavior if it changes.
        return "auto"
    return tool_choice


def reasoning_request_options(settings):
    """Return only options known to be supported by the configured provider."""
    provider = str(settings.get("provider", "")).strip().lower()
    mode = str(settings.get("reasoning_effort", "fast")).strip().lower()
    if mode not in REASONING_MODES:
        mode = "fast"

    # MiMo-V2.5-Pro enables deep thinking by default.  Its OpenAI-compatible
    # API accepts this non-standard field through ``extra_body``.  Detect the
    # model/official endpoint too so existing configs that used the generic or
    # wrong provider label still make the UI's "fast" setting effective.
    if mode == "fast" and _is_mimo_endpoint(settings):
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    # Volcano Engine Ark exposes Doubao's thinking switch through the same
    # OpenAI-compatible ``extra_body`` mechanism.  Keeping it here means both
    # the ASR -> LLM pipeline and the multimodal voice pipeline behave alike.
    # See: https://www.volcengine.com/docs/82379/1795150
    if mode == "fast" and provider in DOUBAO_PROVIDERS:
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    # DashScope exposes Qwen thinking controls through OpenAI extra_body.
    # Omitting the option keeps the model/provider default.
    if mode == "fast" and provider in {"aliyun", "qwen"}:
        return {"extra_body": {"enable_thinking": False}}
    model = str(settings.get("model", "")).strip().lower()
    if mode == "fast" and provider in {"baidu", "baidu_qianfan", "qianfan"}:
        # Qianfan exposes two model-family-specific thinking switches through
        # its OpenAI-compatible Chat Completions API.  Only send a switch to
        # models documented to accept it; unknown/custom endpoints keep their
        # provider defaults instead of receiving an unsupported parameter.
        if model.startswith((
            "ernie-5.0-thinking",
            "ernie-4.5-turbo-vl",
            "ernie-4.5-vl-",
            "qwen3-",
        )):
            return {"extra_body": {"enable_thinking": False}}
        if model.startswith(("deepseek-v3.2", "kimi-k2.5", "glm-5")):
            return {"extra_body": {"thinking": {"type": "disabled"}}}
    zhipu_toggle_models = (
        "glm-4.5",
        "glm-4.7",
        "glm-5",
    )
    if (
        mode == "fast"
        and provider in {"zhipu", "glm"}
        and model.startswith(zhipu_toggle_models)
    ):
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    return {}
