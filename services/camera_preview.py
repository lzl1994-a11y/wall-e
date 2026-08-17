"""On-demand camera preview used by the local configuration page."""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
WORKER_PATH = ROOT / "services" / "camera_preview_worker.py"
DEFAULT_ROS_SETUP = Path("/opt/tros/humble/setup.bash")


class CameraPreview:
    """Run camera capture out of process so blocked drivers can be terminated."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        idle_timeout: float = 15.0,
        frame_rate: float = 8.0,
        startup_timeout: float = 8.0,
        frame_timeout: float = 4.0,
    ) -> None:
        del config_path  # Device ownership belongs to camera_capture_node.
        self._idle_timeout = max(2.0, float(idle_timeout))
        self._frame_rate = max(1.0, float(frame_rate))
        self._startup_timeout = max(0.2, float(startup_timeout))
        self._frame_timeout = max(0.5, float(frame_timeout))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._generation = 0
        self._state = "stopped"
        self._phase = "stopped"
        self._error = ""
        self._diagnostic = ""
        self._device = ""
        self._source = ""
        self._frame: bytes | None = None
        self._frame_time = 0.0
        self._last_client_time = 0.0
        self._width = 0
        self._height = 0
        self._fps = 0.0

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._last_client_time = time.monotonic()
            if self._thread is not None and self._thread.is_alive():
                return self._status_locked()

            self._generation += 1
            generation = self._generation
            self._stop_event = threading.Event()
            self._state = "starting"
            self._phase = "launching"
            self._error = ""
            self._diagnostic = ""
            self._device = ""
            self._source = ""
            self._frame = None
            self._frame_time = 0.0
            self._width = 0
            self._height = 0
            self._fps = 0.0
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(generation, self._stop_event),
                name="camera-preview-supervisor",
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def stop(self, *, wait: bool = True) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            process = self._process
            self._stop_event.set()
            if thread is not None and thread.is_alive():
                self._state = "stopping"
                self._phase = "stopping"
            else:
                self._state = "stopped"
                self._phase = "stopped"
                self._thread = None

        self._terminate_process(process)
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

        with self._lock:
            if thread is None or not thread.is_alive():
                self._state = "stopped"
                self._phase = "stopped"
                self._thread = None
                self._process = None
            return self._status_locked()

    def close(self) -> None:
        self.stop(wait=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def get_frame(self) -> tuple[bytes | None, dict[str, Any]]:
        with self._lock:
            self._last_client_time = time.monotonic()
            frame = self._frame if self._state == "running" else None
            return frame, self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        age_ms = None
        if self._frame_time:
            age_ms = max(0, int((time.monotonic() - self._frame_time) * 1000))
        return {
            "state": self._state,
            "phase": self._phase,
            "device": self._device,
            "source": self._source,
            "width": self._width,
            "height": self._height,
            "fps": round(self._fps, 1),
            "frame_age_ms": age_ms,
            "error": self._error,
            "diagnostic": self._diagnostic,
        }

    def _set_status(self, generation: int, **changes: Any) -> bool:
        with self._lock:
            if generation != self._generation:
                return False
            for key, value in changes.items():
                setattr(self, f"_{key}", value)
            return True

    @staticmethod
    def _ros_python() -> str:
        """Use the launcher's Python unless preview-specific override is configured."""
        configured = os.environ.get("WALI_CAMERA_PREVIEW_PYTHON", "").strip()
        if configured and os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return sys.executable

    def _worker_command(self) -> list[str]:
        command = [
            self._ros_python(),
            str(WORKER_PATH),
            "--fps",
            str(self._frame_rate),
        ]
        # The worker normally inherits the launcher's ROS environment. An
        # explicit override remains available for standalone config_web runs.
        ros_setup = os.environ.get("WALI_CAMERA_PREVIEW_ROS_SETUP", "").strip()
        if not ros_setup and os.name != "nt" and DEFAULT_ROS_SETUP.is_file():
            ros_setup = str(DEFAULT_ROS_SETUP)
        if os.name != "nt" and ros_setup and os.path.isfile(ros_setup):
            return [
                "bash",
                "-c",
                'source "$1" && exec "${@:2}"',
                "camera-preview-worker",
                ros_setup,
                *command,
            ]
        return command

    def _capture_loop(self, generation: int, stop_event: threading.Event) -> None:
        process: subprocess.Popen[str] | None = None
        ended_with_error = False
        messages: queue.Queue[str] = queue.Queue()
        try:
            if not WORKER_PATH.is_file():
                raise RuntimeError(f"摄像头预览采集程序不存在: {WORKER_PATH}")
            if not self._set_status(generation, source="/camera_frame", phase="launching"):
                return

            popen_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.DEVNULL,
                "text": True,
                "encoding": "utf-8",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(self._worker_command(), **popen_kwargs)
            if not self._set_status(generation, process=process, phase="requesting_camera"):
                return

            def read_messages() -> None:
                if process is None or process.stdout is None:
                    return
                for line in process.stdout:
                    messages.put(line)

            reader = threading.Thread(
                target=read_messages,
                name="camera-preview-reader",
                daemon=True,
            )
            reader.start()

            started_at = time.monotonic()
            last_frame_at = 0.0
            while not stop_event.is_set():
                now = time.monotonic()
                with self._lock:
                    if generation != self._generation:
                        return
                    idle = now - self._last_client_time > self._idle_timeout
                if idle:
                    break
                if not last_frame_at and now - started_at > self._startup_timeout:
                    raise RuntimeError("等待 /camera_frame 超时，请检查 camera_capture_node 状态")
                if last_frame_at and now - last_frame_at > self._frame_timeout:
                    raise RuntimeError(
                        f"/camera_frame 画面中断，超过 {self._frame_timeout:g} 秒没有新帧"
                    )

                try:
                    raw_message = messages.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(f"摄像头采集进程已退出，退出码 {return_code}")
                    continue

                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue
                message_type = message.get("type")
                if message_type == "status":
                    phase = str(message.get("phase") or "requesting_camera")
                    changes = {"phase": phase}
                    if message.get("source"):
                        changes["source"] = str(message["source"])
                    if message.get("diagnostic"):
                        changes["diagnostic"] = str(message["diagnostic"])
                    self._set_status(generation, **changes)
                    continue
                if message_type == "error":
                    raise RuntimeError(str(message.get("error") or "摄像头采集失败"))
                if message_type != "frame":
                    continue

                try:
                    frame = base64.b64decode(message["jpeg"], validate=True)
                except (KeyError, ValueError):
                    continue
                if not frame:
                    continue
                last_frame_at = time.monotonic()
                if not self._set_status(
                    generation,
                    state="running",
                    phase="streaming",
                    frame=frame,
                    frame_time=last_frame_at,
                    width=int(message.get("width") or 0),
                    height=int(message.get("height") or 0),
                    fps=float(message.get("fps") or 0.0),
                    source=str(message.get("source") or "/camera_frame"),
                ):
                    return
        except Exception as exc:
            ended_with_error = True
            message = str(exc)
            with self._lock:
                diagnostic = self._diagnostic if generation == self._generation else ""
            if diagnostic and diagnostic not in message:
                message = f"{message}（{diagnostic}）"
            self._set_status(generation, state="error", phase="error", error=message)
            print(f"[CameraPreview] {message}")
        finally:
            self._terminate_process(process)
            with self._lock:
                if generation == self._generation:
                    self._process = None
                    self._thread = None
                    if not ended_with_error:
                        self._state = "stopped"
                        self._phase = "stopped"

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
