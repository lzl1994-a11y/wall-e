"""Provider-specific LLM request options shared by both voice pipelines."""


REASONING_MODES = {"fast", "default"}


def reasoning_request_options(settings):
    """Return only options known to be supported by the configured provider."""
    provider = str(settings.get("provider", "")).strip().lower()
    mode = str(settings.get("reasoning_effort", "fast")).strip().lower()
    if mode not in REASONING_MODES:
        mode = "fast"

    # DashScope exposes Qwen thinking controls through OpenAI extra_body.
    # Omitting the option keeps the model/provider default.
    if mode == "fast" and provider in {"aliyun", "qwen"}:
        return {"extra_body": {"enable_thinking": False}}
    model = str(settings.get("model", "")).strip().lower()
    zhipu_toggle_models = (
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
