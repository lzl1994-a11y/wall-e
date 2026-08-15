"""智谱 ASR 适配器。"""
import requests
from .base import AbstractASR


class ZhipuASR(AbstractASR):
    """智谱语音识别，调用 open.bigmodel.cn 音频转录 API。

    构造时接收 config.yaml asr 节点的 key / url / model。
    模型名必须由配置提供，不在代码中选择默认模型。
    """

    _DEFAULT_URL = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"

    def __init__(self, api_key: str, url: str, model: str):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Zhipu ASR model must be configured in config.yaml")
        self.api_key = api_key
        self.url = url or self._DEFAULT_URL
        self.model = model.strip()

    def recognize(self, wav_path: str, sample_rate: int = 16000) -> str:
        try:
            with open(wav_path, "rb") as f:
                resp = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (wav_path, f, "audio/wav")},
                    data={
                        "model": self.model,
                        "stream": "false",
                    },
                    timeout=15,
                )
            resp.raise_for_status()
            body = resp.json()
            text = (body.get("text") or "").strip()
            return text
        except requests.exceptions.Timeout:
            print("[ZhipuASR] 请求超时")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[ZhipuASR] 请求失败: {e}")
            return ""
        except Exception as e:
            print(f"[ZhipuASR] 识别失败: {e}")
            return ""
