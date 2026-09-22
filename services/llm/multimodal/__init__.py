"""多模态适配器工厂：读 config.llm.provider → 返回对应适配器实例。"""
import yaml
from .aliyun_multimodal import AliyunMultimodal
from .baidu_multimodal import BaiduMultimodal
from .doubao_multimodal import DoubaoMultimodal
from .tencent_hunyuan_multimodal import TencentHunyuanMultimodal
from .xiaomi_mimo_multimodal import XiaomiMiMoMultimodal
from .zhipu_multimodal import ZhipuMultimodal

PROVIDERS = {
    "aliyun": AliyunMultimodal,
    "baidu": BaiduMultimodal,
    "baidu_qianfan": BaiduMultimodal,
    "qianfan": BaiduMultimodal,
    "doubao": DoubaoMultimodal,
    "volcengine": DoubaoMultimodal,
    "ark": DoubaoMultimodal,
    "tencent": TencentHunyuanMultimodal,
    "tencent_hunyuan": TencentHunyuanMultimodal,
    "hunyuan": TencentHunyuanMultimodal,
    "mimo": XiaomiMiMoMultimodal,
    "xiaomi": XiaomiMiMoMultimodal,
    "xiaomi_mimo": XiaomiMiMoMultimodal,
    "zhipu": ZhipuMultimodal,
}


def create_multimodal(config_path: str = "core/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    provider = config["llm"]["provider"]
    return PROVIDERS[provider]()
