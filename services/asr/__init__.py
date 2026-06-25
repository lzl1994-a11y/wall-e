"""ASR 适配器工厂：读 config.asr.provider → 返回对应适配器实例。"""
import yaml

from .aliyun_asr import AliyunASR
from .zhipu_asr import ZhipuASR

PROVIDERS = {
    "aliyun": AliyunASR,
    "zhipu": ZhipuASR,
}


def create_asr(config_path: str = "core/config.yaml"):
    """创建 ASR 适配器实例。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    asr_cfg = config["asr"]
    provider = asr_cfg["provider"]
    cls = PROVIDERS[provider]

    kwargs = {"api_key": asr_cfg["key"], "model": asr_cfg["model"]}
    if "url" in asr_cfg:
        kwargs["url"] = asr_cfg["url"]
    return cls(**kwargs)
