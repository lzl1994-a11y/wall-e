"""Small per-launcher readiness markers and offline startup prompt state.

Only the launcher creates the session directory. Missing state disables the
feature, so standalone nodes keep their existing behaviour. No ROS, devices or
network clients are imported here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from uuid import uuid4


STARTUP_STATUS_ENV = "WALI_STARTUP_STATUS_DIR"
STARTUP_REQUIRED_ENV = "WALI_STARTUP_REQUIRED"
STARTUP_COMPONENT_ENV = "WALI_STARTUP_COMPONENT"
PROMPT_TIMEOUT_SECONDS = 15.0
POST_PROMPT_GUARD_SECONDS = 0.35
STARTUP_PROMPT_ASSETS = {
    "startup_begin": "startup_begin.wav",
    "startup_ready": "startup_ready.wav",
}
_COMPONENT = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_ACTIVE_GATE = "prompt-active.json"


def _future_deadline(value: Any, now: float, maximum: float) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and 0 < value - now <= maximum + 0.01)


def _session_directory(directory: Path | str | None = None) -> Path | None:
    value = directory if directory is not None else os.environ.get(STARTUP_STATUS_ENV)
    if not value:
        return None
    try:
        root = Path(value)
        return root if root.is_dir() else None
    except (OSError, TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _pid_alive(pid: Any) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # os.kill(pid, 0) does not have POSIX probing semantics on Windows.
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            status = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(status))) and status.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    if Path("/proc").is_dir():
        try:
            # kill(pid, 0) also succeeds for unreaped zombie children.
            state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                return False
        except FileNotFoundError:
            return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def mark_startup_ready(component: str | None = None) -> bool:
    """Publish one component's readiness into the current launcher session."""
    root = _session_directory()
    name = component if component is not None else os.environ.get(STARTUP_COMPONENT_ENV)
    if root is None or not isinstance(name, str) or not _COMPONENT.fullmatch(name):
        return False
    try:
        _write_json(root / f"{name}.ready", {"pid": os.getpid()})
        return True
    except OSError:
        return False


def clear_startup_ready(component: str | None = None, directory: Path | str | None = None) -> None:
    """Invalidate readiness before the launcher creates or restarts an owner."""
    root = _session_directory(directory)
    name = component if component is not None else os.environ.get(STARTUP_COMPONENT_ENV)
    if root is None or not isinstance(name, str) or not _COMPONENT.fullmatch(name):
        return
    try:
        (root / f"{name}.ready").unlink(missing_ok=True)
    except OSError:
        pass


class StartupPromptGate:
    """Pause recognition only while a startup clip or its acoustic tail plays."""

    def __init__(self, directory: Path | str | None = None):
        self.directory = _session_directory(directory)
        self._finished = self.directory is None

    def active(self) -> bool:
        if self._finished:
            return False
        now = time.monotonic()
        finished = _read_json(self.directory / "startup_ready.finished")
        if finished is not None:
            release = finished.get("release_at", 0)
            if _future_deadline(release, now, POST_PROMPT_GUARD_SECONDS):
                return True
            self._finished = True
            return False
        gate = _read_json(self.directory / _ACTIVE_GATE)
        if gate is None:
            return False
        deadline = gate.get("deadline")
        if not _future_deadline(deadline, now, PROMPT_TIMEOUT_SECONDS):
            return False
        # A dead playback process must never leave recognition muted.
        return gate.get("phase") == "tail" or _pid_alive(gate.get("pid"))


class StartupPromptController:
    """Issue each clip once, with ready tied to live component acknowledgements."""

    def __init__(self, directory: Path | str | None = None, assets_root: Path | str | None = None):
        self.directory = _session_directory(directory)
        self._lock = threading.RLock()
        self._finished = False
        self._pending_request: str | None = None
        self._pending_cue: str | None = None
        self._clips: dict[str, bytes] = {}
        self.required: tuple[str, ...] = ()
        if self.directory is None:
            return
        try:
            required = json.loads(os.environ.get(STARTUP_REQUIRED_ENV, "[]"))
        except ValueError:
            required = []
        if (not isinstance(required, list) or not required
                or any(not isinstance(name, str) or not _COMPONENT.fullmatch(name) for name in required)):
            self.directory = None
            return
        self.required = tuple(dict.fromkeys(required))
        root = Path(assets_root) if assets_root is not None else Path(__file__).resolve().parents[2] / "assets"
        from services.audio.esp32_network_prompt import load_prompt_pcm
        self._clips = {cue: load_prompt_pcm(root / filename) for cue, filename in STARTUP_PROMPT_ASSETS.items()}

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    @property
    def finished(self) -> bool:
        with self._lock:
            if self.directory is not None and (self.directory / "startup_ready.finished").exists():
                self._finished = True
            return self._finished

    def skip_ready(self) -> None:
        """Retire optional feedback once direct-audio conversation has begun."""
        if not self.enabled:
            return
        with self._lock:
            if self._pending_request:
                self.cancel(self._pending_request)
            try:
                if not (self.directory / "startup_ready.finished").exists():
                    request_id = f"startup_ready-skipped-{uuid4().hex}"
                    _write_json(self.directory / "startup_ready.issued", {
                        "request_id": request_id, "pid": os.getpid(),
                    })
                    _write_json(self.directory / "startup_ready.finished", {
                        "request_id": request_id, "cancelled": True,
                        "release_at": time.monotonic() + POST_PROMPT_GUARD_SECONDS,
                    })
            except OSError:
                pass
            self._finished = True

    @property
    def pending_request(self) -> str | None:
        with self._lock:
            return self._pending_request

    @property
    def pending_cue(self) -> str | None:
        with self._lock:
            return self._pending_cue

    def _all_ready(self) -> bool:
        return all(_pid_alive((_read_json(self.directory / f"{name}.ready") or {}).get("pid")) for name in self.required)

    def next_prompt(self) -> tuple[str, str, bytes] | None:
        if not self.enabled or self._finished:
            return None
        with self._lock:
            if (self.directory / "stopping").exists():
                return None
            if (self.directory / "startup_ready.finished").exists():
                self._finished = True
                return None
            if StartupPromptGate(self.directory).active():
                return None
            cue = "startup_begin"
            if (self.directory / "startup_begin.issued").exists():
                cue = "startup_ready"
                if (not (self.directory / "startup_begin.finished").exists()
                        or (self.directory / "startup_ready.issued").exists() or not self._all_ready()):
                    return None
            request_id = f"{cue}-{uuid4().hex}"
            value = {"request_id": request_id, "pid": os.getpid()}
            claim = self.directory / f"{cue}.issued"
            try:
                # Exclusive creation keeps node restarts from replaying a cue.
                with claim.open("x", encoding="utf-8") as target:
                    json.dump(value, target, separators=(",", ":"))
                _write_json(self.directory / _ACTIVE_GATE, {
                    **value, "phase": "playing", "deadline": time.monotonic() + PROMPT_TIMEOUT_SECONDS,
                })
            except FileExistsError:
                return None
            except OSError:
                return None
            self._pending_request = request_id
            self._pending_cue = cue
            return request_id, cue, self._clips[cue]

    def finish(self, request_id: str) -> bool:
        """Release this request's gate without overwriting a newer prompt."""
        return self._complete(request_id, cancelled=False)

    def cancel(self, request_id: str) -> bool:
        """Yield to foreground audio and keep issued cues from replaying."""
        return self._complete(request_id, cancelled=True)

    def _complete(self, request_id: str, *, cancelled: bool) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            for cue in STARTUP_PROMPT_ASSETS:
                issued = _read_json(self.directory / f"{cue}.issued")
                if issued is None or issued.get("request_id") != request_id:
                    continue
                if self._pending_request == request_id:
                    self._pending_request = self._pending_cue = None
                if (self.directory / f"{cue}.finished").exists():
                    return False
                release = time.monotonic() + POST_PROMPT_GUARD_SECONDS
                try:
                    _write_json(self.directory / f"{cue}.finished", {
                        "request_id": request_id, "release_at": release, "cancelled": cancelled,
                    })
                    gate = _read_json(self.directory / _ACTIVE_GATE)
                    if gate is not None and gate.get("request_id") == request_id:
                        _write_json(self.directory / _ACTIVE_GATE, {
                            "request_id": request_id, "pid": os.getpid(),
                            "phase": "tail", "deadline": release,
                        })
                    return True
                except OSError:
                    return False
        return False
