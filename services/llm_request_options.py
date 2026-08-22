"""Provider-specific LLM request options shared by both voice pipelines."""


REASONING_MODES = {"fast", "default"}
DOUBAO_PROVIDERS = {"doubao", "volcengine", "ark"}


def reasoning_request_options(settings):
    """Return only options known to be supported by the configured provider."""
    provider = str(settings.get("provider", "")).strip().lower()
    mode = str(settings.get("reasoning_effort", "fast")).strip().lower()
    if mode not in REASONING_MODES:
        mode = "fast"

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
