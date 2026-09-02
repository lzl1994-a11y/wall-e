"""Opt-in rolling capture of voice-pipeline debug artifacts."""

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path


VOICE_DEBUG_ENV = "WALI_SAVE_VOICE_DEBUG"
VOICE_DEBUG_LIMIT = 20


def voice_debug_enabled() -> bool:
    return os.environ.get(VOICE_DEBUG_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RollingVoiceDebugStore:
    """Save artifacts only when enabled and retain the newest files per group."""

    def __init__(self, enabled=None, root=None, limit=VOICE_DEBUG_LIMIT):
        self.enabled = voice_debug_enabled() if enabled is None else bool(enabled)
        self.root = Path(root or "~/.wali_debug/voice").expanduser()
        self.limit = int(limit)
        self._lock = threading.Lock()

    def save_file(self, group: str, source, suffix=None):
        if not self.enabled:
            return None
        source_path = Path(source)
        target = self._new_path(group, suffix or source_path.suffix)
        try:
            with self._lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
                self._prune(target.parent)
        except OSError as exc:
            print(f"[VoiceDebug] 保存 {group} 失败: {exc}")
            return None
        return target

    def save_json(self, group: str, payload):
        if not self.enabled:
            return None
        target = self._new_path(group, ".json")
        try:
            with self._lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
                    stream.write("\n")
                self._prune(target.parent)
        except OSError as exc:
            print(f"[VoiceDebug] 保存 {group} 失败: {exc}")
            return None
        return target

    def _new_path(self, group: str, suffix: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique = f"{time.time_ns() % 1_000_000_000:09d}-{uuid.uuid4().hex[:6]}"
        return self.root / group / f"{timestamp}-{unique}{suffix}"

    def _prune(self, directory: Path):
        files = sorted(
            (path for path in directory.iterdir() if path.is_file()),
            key=lambda path: path.name,
            reverse=True,
        )
        for stale in files[self.limit:]:
            try:
                stale.unlink()
            except OSError:
                pass
