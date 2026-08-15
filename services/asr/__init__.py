"""ASR 适配器工厂：根据配置选择一个云端或本地识别引擎。"""

from pathlib import Path

import yaml

from .aliyun_asr import AliyunASR
from .baidu_asr import BaiduASR
from .zhipu_asr import ZhipuASR

PROVIDERS = {
    "aliyun": AliyunASR,
    "baidu": BaiduASR,
    "zhipu": ZhipuASR,
}

LOCAL_ENGINES = {
    "sherpa_onnx_zipformer",
    "sherpa_onnx_paraformer",
    "sherpa_onnx_sensevoice",
    "sherpa_onnx_whisper",
    "faster_whisper",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_path(value: object, field: str, *, directory: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be configured in config.yaml")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{field} {kind} does not exist: {path}")
    return str(path)


def _num_threads(settings: dict, engine: str) -> int:
    value = settings.get("num_threads", 2)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
        raise ValueError(f"asr.{engine}.num_threads must be an integer from 1 to 64")
    return value


def _setting_string(settings: dict, engine: str, key: str, default: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"asr.{engine}.{key} must be a non-empty string")
    return value.strip()


def _setting_choice(
    settings: dict,
    engine: str,
    key: str,
    default: str,
    choices: set[str],
) -> str:
    value = _setting_string(settings, engine, key, default)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"asr.{engine}.{key} must be one of: {allowed}")
    return value


def _create_local_asr(asr_cfg: dict):
    engine = asr_cfg.get("engine")
    if engine not in LOCAL_ENGINES:
        raise ValueError(f"Unsupported local ASR engine: {engine}")
    settings = asr_cfg.get(engine)
    if not isinstance(settings, dict):
        raise ValueError(f"asr.{engine} must be configured in config.yaml")

    if engine == "faster_whisper":
        from .faster_whisper_asr import FasterWhisperASR

        return FasterWhisperASR(
            model_path=_model_path(
                settings.get("model_path"),
                f"asr.{engine}.model_path",
                directory=True,
            ),
            language=_setting_string(settings, engine, "language", "zh"),
            device=_setting_choice(
                settings,
                engine,
                "device",
                "cpu",
                {"cpu", "cuda", "auto"},
            ),
            compute_type=_setting_choice(
                settings,
                engine,
                "compute_type",
                "int8",
                {
                    "default",
                    "int8",
                    "int8_float16",
                    "int8_float32",
                    "float16",
                    "float32",
                },
            ),
        )

    from .sherpa_onnx_asr import (
        SherpaParaformerASR,
        SherpaSenseVoiceASR,
        SherpaWhisperASR,
        SherpaZipformerASR,
    )

    num_threads = _num_threads(settings, engine)
    if engine == "sherpa_onnx_zipformer":
        return SherpaZipformerASR(
            encoder=_model_path(settings.get("encoder"), f"asr.{engine}.encoder"),
            decoder=_model_path(settings.get("decoder"), f"asr.{engine}.decoder"),
            joiner=_model_path(settings.get("joiner"), f"asr.{engine}.joiner"),
            tokens=_model_path(settings.get("tokens"), f"asr.{engine}.tokens"),
            num_threads=num_threads,
        )
    if engine == "sherpa_onnx_paraformer":
        return SherpaParaformerASR(
            model=_model_path(settings.get("model"), f"asr.{engine}.model"),
            tokens=_model_path(settings.get("tokens"), f"asr.{engine}.tokens"),
            num_threads=num_threads,
        )
    if engine == "sherpa_onnx_sensevoice":
        use_itn = settings.get("use_itn", True)
        if not isinstance(use_itn, bool):
            raise ValueError(f"asr.{engine}.use_itn must be a boolean")
        return SherpaSenseVoiceASR(
            model=_model_path(settings.get("model"), f"asr.{engine}.model"),
            tokens=_model_path(settings.get("tokens"), f"asr.{engine}.tokens"),
            language=_setting_string(settings, engine, "language", "auto"),
            use_itn=use_itn,
            num_threads=num_threads,
        )
    return SherpaWhisperASR(
        encoder=_model_path(settings.get("encoder"), f"asr.{engine}.encoder"),
        decoder=_model_path(settings.get("decoder"), f"asr.{engine}.decoder"),
        tokens=_model_path(settings.get("tokens"), f"asr.{engine}.tokens"),
        language=_setting_string(settings, engine, "language", "zh"),
        num_threads=num_threads,
    )


def create_asr(config_path: str = "core/config.yaml"):
    """创建 ASR 适配器实例。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not isinstance(config.get("asr"), dict):
        raise ValueError("asr must be configured in config.yaml")
    asr_cfg = config["asr"]
    mode = asr_cfg.get("mode", asr_cfg.get("type", "cloud"))
    if mode == "local":
        return _create_local_asr(asr_cfg)
    if mode != "cloud":
        raise ValueError(f"Unsupported ASR mode: {mode}")

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

    model = provider_cfg.get("model") or asr_cfg.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"asr.{provider}.model must be configured in config.yaml")
    model = model.strip()
    if provider == "aliyun":
        return AliyunASR(api_key=api_key, model=model)
    if provider == "zhipu":
        return ZhipuASR(
            api_key=api_key,
            model=model,
            url=provider_cfg.get("url") or asr_cfg.get("url", ""),
        )
    raise ValueError(f"Unsupported ASR provider: {provider}")
