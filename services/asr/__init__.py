"""ASR 适配器工厂：读 config.asr.provider → 返回对应适配器实例。"""
import yaml

from .aliyun_asr import AliyunASR
from .baidu_asr import BaiduASR
from .zhipu_asr import ZhipuASR

PROVIDERS = {
    "aliyun": AliyunASR,
    "baidu": BaiduASR,
    "zhipu": ZhipuASR,
}


def create_asr(config_path: str = "core/config.yaml"):
    """创建 ASR 适配器实例。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    asr_cfg = config["asr"]
    provider = asr_cfg["provider"]
    provider_cfg = asr_cfg.get(provider)
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}

    # Nested provider settings are preferred; flat keys keep old configs valid.
    api_key = provider_cfg.get("api_key") or asr_cfg.get("key", "")
    if provider == "baidu":
        return BaiduASR(
            app_id=provider_cfg["app_id"],
            api_key=api_key,
            dev_pid=provider_cfg.get("dev_pid", 15372),
            cuid=provider_cfg.get("cuid", "wali-x3"),
            url=provider_cfg.get("url", BaiduASR.DEFAULT_URL),
            lm_id=provider_cfg.get("lm_id"),
            user=provider_cfg.get("user"),
        )

    model = provider_cfg.get("model") or asr_cfg.get("model", "")
    if provider == "aliyun":
        return AliyunASR(api_key=api_key, model=model)
    if provider == "zhipu":
        return ZhipuASR(
            api_key=api_key,
            model=model,
            url=provider_cfg.get("url") or asr_cfg.get("url", ""),
        )
    raise ValueError(f"Unsupported ASR provider: {provider}")
