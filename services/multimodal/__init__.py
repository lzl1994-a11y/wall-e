"""多模态适配器工厂：读 config.llm.provider → 返回对应适配器实例。"""
import yaml
from .aliyun_multimodal import AliyunMultimodal
from .zhipu_multimodal import ZhipuMultimodal

PROVIDERS = {
    "aliyun": AliyunMultimodal,
    "zhipu": ZhipuMultimodal,
}


def create_multimodal(config_path: str = "core/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    provider = config["llm"]["provider"]
    return PROVIDERS[provider]()
