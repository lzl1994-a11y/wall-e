"""Author, validate, and persist deterministic robot choreographies.

The editor works with relative, normalized motion segments because they are
easy to author.  This service compiles them from a known neutral pose into
absolute PWM endpoints before they can be used by a runtime.
"""

from __future__ import annotations

import math
import mimetypes
import re
import threading
import uuid
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from services.motion.sequence_execution import SequenceLibrary
from services.motion.servo_motion_config import neck_kinematics_from_servos


SCHEMA_VERSION = 1
MAX_TIMELINE_SECONDS = 600.0
MAX_SEGMENTS = 500
MAX_AUDIO_BYTES = 64 * 1024 * 1024
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
AUDIO_ID_PATTERN = re.compile(r"^[a-f0-9]{16}\.(?:mp3|wav|ogg|m4a|aac|flac)$")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
MOTOR_DEFINITIONS = (
    ("chassis", "履带", ("forward", "backward", "left", "right", "spin")),
)

CHANNEL_DEFINITIONS = (
    ("head_yaw", "头部左右", ("head_yaw",)),
    ("neck_pitch", "脖子俯仰", ("neck_top", "neck_bottom")),
    ("eye_l", "左眼", ("eye_l",)),
    ("eye_r", "右眼", ("eye_r",)),
    ("arm_l", "左臂", ("arm_l",)),
    ("arm_r", "右臂", ("arm_r",)),
    ("eyebrow_l", "左眉", ("eyebrow_l",)),
    ("eyebrow_r", "右眉", ("eyebrow_r",)),
)


class ChoreographyError(ValueError):
    """A choreography cannot be loaded, validated, or saved."""

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _round_time(value: float) -> float:
    return round(float(value), 3)


class ChoreographyStore:
    """Filesystem store plus compiler used by both Web and future ROS adapters."""

    def __init__(
        self,
        config_path: Path | str,
        *,
        directory: Path | str | None = None,
        sequences_path: Path | str | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.directory = Path(
            directory or self.config_path.parent / "choreographies"
        ).expanduser().resolve()
        self.sequences_path = Path(
            sequences_path or self.config_path.parent / "sequences.yaml"
        ).expanduser().resolve()
        self.media_directory = self.directory / "media"
        self._lock = threading.RLock()

    def _load_yaml_mapping(self, path: Path) -> dict[str, Any]:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            return {}
        except (OSError, yaml.YAMLError) as exc:
            raise ChoreographyError(f"无法读取 {path.name}: {exc}") from exc
        if not isinstance(value, dict):
            raise ChoreographyError(f"{path.name} 顶层必须是对象")
        return value

    def _servo_config(self) -> dict[str, dict[str, Any]]:
        config = self._load_yaml_mapping(self.config_path)
        return {
            item["name"]: item
            for item in config.get("servos", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }

    def channel_catalog(self) -> list[dict[str, Any]]:
        servos = self._servo_config()
        result = []
        for channel, label, members in CHANNEL_DEFINITIONS:
            if all(member in servos for member in members):
                result.append({
                    "id": channel,
                    "label": label,
                    "members": list(members),
                    "minimum": -100,
                    "maximum": 100,
                })
        known_members = {member for _, _, members in CHANNEL_DEFINITIONS for member in members}
        for name in sorted(servos):
            if name not in known_members:
                result.append({
                    "id": name,
                    "label": name,
                    "members": [name],
                    "minimum": -100,
                    "maximum": 100,
                })
        return result

    def _sequence_catalog_data(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        payload = self._load_yaml_mapping(self.sequences_path)
        sequences = payload.get("sequences", {})
        poses = payload.get("poses", {})
        metadata = payload.get("action_metadata", {})
        return (
            sequences if isinstance(sequences, dict) else {},
            poses if isinstance(poses, dict) else {},
            metadata if isinstance(metadata, dict) else {},
        )

    @staticmethod
    def _action_metadata(metadata: Mapping[str, Any], name: str) -> dict[str, str]:
        value = metadata.get(name, {})
        if not isinstance(value, Mapping):
            value = {}
        label = value.get("label")
        remark = value.get("remark")
        return {
            "label": str(label).strip() if isinstance(label, str) and label.strip() else name.replace("_", " "),
            "remark": str(remark).strip() if isinstance(remark, str) else "",
        }

    @staticmethod
    def _motor_catalog() -> list[dict[str, Any]]:
        return [
            {
                "id": motor,
                "label": label,
                "directions": list(directions),
            }
            for motor, label, directions in MOTOR_DEFINITIONS
        ]

    @staticmethod
    def _channels_for_frames(
        frames: list[dict[str, Any]], poses: Mapping[str, Any]
    ) -> list[str]:
        channels: set[str] = set()
        for frame in frames:
            for action in frame.get("actions", []):
                action_type = action.get("type")
                if action_type == "servo" and isinstance(action.get("name"), str):
                    channels.add(action["name"])
                elif action_type == "pose":
                    pose = poses.get(action.get("name"), {})
                    if isinstance(pose, dict) and isinstance(pose.get("targets"), dict):
                        channels.update(str(name) for name in pose["targets"])
                elif action_type == "manual_servo" and isinstance(action.get("targets"), dict):
                    channels.update(str(name) for name in action["targets"])
                elif action_type == "motor":
                    channels.add("chassis")
        return sorted(channels)

    def action_catalog(self) -> list[dict[str, Any]]:
        sequences, poses, metadata = self._sequence_catalog_data()
        library = SequenceLibrary(sequences, poses)
        catalog = []
        for name in sorted(set(sequences) | set(poses)):
            frames = library.flatten(name)
            if not frames:
                continue
            duration = 0.6
            for frame in frames:
                timestamp = float(frame.get("time", 0.0))
                duration = max(duration, timestamp + 0.25)
                for action in frame.get("actions", []):
                    if action.get("type") == "motor" and _finite_number(action.get("duration")):
                        duration = max(duration, timestamp + float(action["duration"]))
            item = {
                "id": name,
                "duration": _round_time(duration),
                "channels": self._channels_for_frames(frames, poses),
            }
            item.update(self._action_metadata(metadata, name))
            catalog.append(item)
        return catalog

    def catalog(self) -> dict[str, Any]:
        return {
            "channels": self.channel_catalog(),
            "motors": self._motor_catalog(),
            "actions": self.action_catalog(),
            "audio": self.list_audio(),
        }

    def list_audio(self) -> list[dict[str, Any]]:
        if not self.media_directory.exists():
            return []
        assets = []
        for metadata_path in sorted(self.media_directory.glob("*.meta.yaml")):
            try:
                metadata = self._load_yaml_mapping(metadata_path)
                asset_id = metadata.get("id")
                if not isinstance(asset_id, str) or not AUDIO_ID_PATTERN.fullmatch(asset_id):
                    continue
                if not (self.media_directory / asset_id).is_file():
                    continue
                assets.append(metadata)
            except ChoreographyError:
                continue
        return assets

    def upload_audio(
        self,
        filename: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        duration: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(filename, str) or not filename.strip() or len(filename) > 200:
            raise ChoreographyError("音乐文件名无效")
        if Path(filename).name != filename or any(char in filename for char in ("/", "\\", "\x00")):
            raise ChoreographyError("音乐文件名不能包含路径")
        suffix = Path(filename).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise ChoreographyError("只支持 MP3、WAV、OGG、M4A、AAC 和 FLAC 音频")
        if not body or len(body) > MAX_AUDIO_BYTES:
            raise ChoreographyError("音乐文件不能为空或超过 64MB")
        if not _finite_number(duration) or not 0.1 <= float(duration) <= MAX_TIMELINE_SECONDS:
            raise ChoreographyError(
                f"音乐时长必须为 0.1–{int(MAX_TIMELINE_SECONDS)} 秒"
            )
        asset_id = f"{uuid.uuid4().hex[:16]}{suffix}"
        resolved_type = (
            content_type.split(";", 1)[0].strip()
            if isinstance(content_type, str) and content_type.startswith("audio/")
            else mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        metadata = {
            "id": asset_id,
            "name": filename,
            "content_type": resolved_type,
            "size": len(body),
            "duration": _round_time(float(duration)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self.media_directory.mkdir(parents=True, exist_ok=True)
            audio_path = self.media_directory / asset_id
            metadata_path = self.media_directory / f"{asset_id}.meta.yaml"
            audio_temp = self.media_directory / f".{asset_id}.{uuid.uuid4().hex}.tmp"
            metadata_temp = self.media_directory / f".{asset_id}.{uuid.uuid4().hex}.meta.tmp"
            try:
                audio_temp.write_bytes(body)
                metadata_temp.write_text(
                    yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                audio_temp.replace(audio_path)
                metadata_temp.replace(metadata_path)
            except OSError as exc:
                audio_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise ChoreographyError(f"保存音乐失败: {exc}") from exc
            finally:
                audio_temp.unlink(missing_ok=True)
                metadata_temp.unlink(missing_ok=True)
        return metadata

    def get_audio(self, asset_id: str) -> tuple[bytes, str]:
        if not isinstance(asset_id, str) or not AUDIO_ID_PATTERN.fullmatch(asset_id):
            raise ChoreographyError("音乐资源 ID 无效")
        path = (self.media_directory / asset_id).resolve()
        if path.parent != self.media_directory.resolve() or not path.is_file():
            raise ChoreographyError("音乐资源不存在")
        metadata_path = self.media_directory / f"{asset_id}.meta.yaml"
        metadata = self._load_yaml_mapping(metadata_path)
        try:
            return path.read_bytes(), str(metadata.get("content_type") or "application/octet-stream")
        except OSError as exc:
            raise ChoreographyError(f"无法读取音乐资源: {exc}") from exc

    def list(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        items = []
        for path in sorted(self.directory.glob("*.yaml")):
            try:
                document = self._load_yaml_mapping(path)
            except ChoreographyError:
                continue
            items.append({
                "id": path.stem,
                "name": str(document.get("name") or path.stem),
                "timeline_seconds": document.get("timeline_seconds", 8),
                "modified_at": path.stat().st_mtime,
            })
        return items

    def _path_for(self, choreography_id: str) -> Path:
        if not isinstance(choreography_id, str) or not ID_PATTERN.fullmatch(choreography_id):
            raise ChoreographyError("动作 ID 只能使用小写字母、数字、下划线和短横线")
        path = (self.directory / f"{choreography_id}.yaml").resolve()
        if path.parent != self.directory:
            raise ChoreographyError("动作文件路径无效")
        return path

    def get(self, choreography_id: str) -> dict[str, Any]:
        path = self._path_for(choreography_id)
        try:
            document = self._load_yaml_mapping(path)
        except ChoreographyError:
            raise
        if not path.is_file():
            raise ChoreographyError("动作编排不存在")
        return document

    def delete(self, choreography_id: str) -> None:
        path = self._path_for(choreography_id)
        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError as exc:
                raise ChoreographyError("动作编排不存在") from exc
            except OSError as exc:
                raise ChoreographyError(f"删除动作编排失败: {exc}") from exc

    def save(self, document: Mapping[str, Any]) -> dict[str, Any]:
        normalized, compiled = self.validate(document)
        path = self._path_for(normalized["id"])
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                temporary.write_text(
                    yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                temporary.replace(path)
            except OSError as exc:
                raise ChoreographyError(f"保存动作编排失败: {exc}") from exc
            finally:
                if "temporary" in locals() and temporary.exists():
                    temporary.unlink(missing_ok=True)
        return {"document": normalized, "compiled": compiled}

    def validate(
        self, document: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(document, Mapping):
            raise ChoreographyError("动作编排必须是对象")
        errors: list[str] = []
        choreography_id = document.get("id")
        if not isinstance(choreography_id, str) or not ID_PATTERN.fullmatch(choreography_id):
            errors.append("动作 ID 必须以小写字母开头，只能包含小写字母、数字、下划线和短横线")
        name = document.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            errors.append("动作名称必须为 1–80 个字符")
        remark = document.get("remark", "")
        if not isinstance(remark, str) or len(remark.strip()) > 200:
            errors.append("动作备注不能超过 200 个字符")
            remark = ""
        remark = remark.strip() if isinstance(remark, str) else ""
        timeline = document.get("timeline_seconds")
        if not _finite_number(timeline) or not 1 <= float(timeline) <= MAX_TIMELINE_SECONDS:
            errors.append(f"时间轴长度必须为 1–{int(MAX_TIMELINE_SECONDS)} 秒")
            timeline = 8.0
        timeline = float(timeline)

        catalog = {item["id"]: item for item in self.channel_catalog()}
        raw_tracks = document.get("tracks", [])
        if not isinstance(raw_tracks, list):
            errors.append("tracks 必须是数组")
            raw_tracks = []
        # Accept the compact legacy/editor form that stores the chassis row in
        # ``tracks``.  New documents use the clearer ``motor_tracks`` field,
        # but accepting both keeps hand-authored YAML easy to migrate.
        chassis_tracks = [
            {
                "motor": "chassis",
                "segments": raw_track.get("segments", []),
            }
            for raw_track in raw_tracks
            if isinstance(raw_track, Mapping) and raw_track.get("channel") == "chassis"
        ]
        raw_tracks = [
            raw_track
            for raw_track in raw_tracks
            if not (isinstance(raw_track, Mapping) and raw_track.get("channel") == "chassis")
        ]
        seen_tracks: set[str] = set()
        normalized_tracks = []
        compiled_transitions = []
        resource_intervals: list[dict[str, Any]] = []
        segment_count = 0
        servos = self._servo_config()

        for track_index, raw_track in enumerate(raw_tracks):
            if not isinstance(raw_track, Mapping):
                errors.append(f"轨道 {track_index + 1} 格式无效")
                continue
            channel = raw_track.get("channel")
            if channel not in catalog:
                errors.append(f"未知逻辑机构: {channel}")
                continue
            if channel in seen_tracks:
                errors.append(f"逻辑机构 {channel} 存在重复轨道")
                continue
            seen_tracks.add(str(channel))
            segments = raw_track.get("segments", [])
            if not isinstance(segments, list):
                errors.append(f"{catalog[channel]['label']} 的动作段必须是数组")
                continue

            parsed_segments = []
            for segment_index, raw_segment in enumerate(segments):
                segment_count += 1
                if segment_count > MAX_SEGMENTS:
                    errors.append(f"动作段不能超过 {MAX_SEGMENTS} 个")
                    break
                prefix = f"{catalog[channel]['label']} 第 {segment_index + 1} 段"
                if not isinstance(raw_segment, Mapping):
                    errors.append(f"{prefix}格式无效")
                    continue
                start = raw_segment.get("start")
                duration = raw_segment.get("duration")
                delta = raw_segment.get("delta")
                if not _finite_number(start) or float(start) < 0:
                    errors.append(f"{prefix}开始时间必须大于等于 0")
                    continue
                if not _finite_number(duration) or not 0.05 <= float(duration) <= 30:
                    errors.append(f"{prefix}持续时间必须为 0.05–30 秒")
                    continue
                if not _finite_number(delta) or not -100 <= float(delta) <= 100 or float(delta) == 0:
                    errors.append(f"{prefix}位移必须为 -100% 到 100%，且不能为 0")
                    continue
                start = _round_time(float(start))
                duration = _round_time(float(duration))
                delta = round(float(delta), 2)
                if start + duration > timeline + 1e-6:
                    errors.append(f"{prefix}超出时间轴范围")
                    continue
                parsed_segments.append({
                    "id": str(raw_segment.get("id") or uuid.uuid4().hex[:10]),
                    "start": start,
                    "duration": duration,
                    "delta": delta,
                })

            parsed_segments.sort(key=lambda item: (item["start"], item["id"]))
            planned_position = 0.0
            previous_end = 0.0
            for segment in parsed_segments:
                if segment["start"] < previous_end - 1e-6:
                    errors.append(f"{catalog[channel]['label']} 的动作段发生时间重叠")
                previous_end = max(previous_end, segment["start"] + segment["duration"])
                end_position = planned_position + segment["delta"]
                if not -100 <= end_position <= 100:
                    errors.append(
                        f"{catalog[channel]['label']} 在 {segment['start']:g}s 后将达到 "
                        f"{end_position:g}%，超出 -100%～100%"
                    )
                transition = {
                    "channel": channel,
                    "start": segment["start"],
                    "duration": segment["duration"],
                    "from_percent": round(planned_position, 2),
                    "to_percent": round(end_position, 2),
                    "delta_percent": segment["delta"],
                }
                if -100 <= planned_position <= 100 and -100 <= end_position <= 100:
                    transition["from_targets"] = self._targets_for(
                        str(channel), planned_position, servos
                    )
                    transition["to_targets"] = self._targets_for(
                        str(channel), end_position, servos
                    )
                compiled_transitions.append(transition)
                for resource in catalog[channel]["members"]:
                    resource_intervals.append({
                        "resource": resource,
                        "start": segment["start"],
                        "end": segment["start"] + segment["duration"],
                        "kind": "segment",
                        "label": f"{catalog[channel]['label']}动作段",
                    })
                planned_position = end_position
            normalized_tracks.append({"channel": channel, "segments": parsed_segments})

        motor_catalog = {
            item["id"]: item for item in self._motor_catalog()
        }
        raw_motor_tracks = document.get("motor_tracks", [])
        if not isinstance(raw_motor_tracks, list):
            errors.append("motor_tracks 必须是数组")
            raw_motor_tracks = []
        raw_motor_tracks = chassis_tracks + raw_motor_tracks
        seen_motors: set[str] = set()
        normalized_motor_tracks = []
        for track_index, raw_track in enumerate(raw_motor_tracks):
            if not isinstance(raw_track, Mapping):
                errors.append(f"履带轨道 {track_index + 1} 格式无效")
                continue
            motor = raw_track.get("motor")
            definition = motor_catalog.get(motor)
            if definition is None:
                errors.append(f"未知电机轨道: {motor}")
                continue
            if motor in seen_motors:
                errors.append(f"电机 {motor} 存在重复轨道")
                continue
            seen_motors.add(str(motor))
            segments = raw_track.get("segments", [])
            if not isinstance(segments, list):
                errors.append(f"{definition['label']} 的动作段必须是数组")
                continue
            parsed_segments = []
            for segment_index, raw_segment in enumerate(segments):
                segment_count += 1
                if segment_count > MAX_SEGMENTS:
                    errors.append(f"动作段不能超过 {MAX_SEGMENTS} 个")
                    break
                prefix = f"{definition['label']} 第 {segment_index + 1} 段"
                if not isinstance(raw_segment, Mapping):
                    errors.append(f"{prefix}格式无效")
                    continue
                start = raw_segment.get("start")
                duration = raw_segment.get("duration")
                direction = raw_segment.get("direction")
                if not _finite_number(start) or float(start) < 0:
                    errors.append(f"{prefix}开始时间必须大于等于 0")
                    continue
                if not _finite_number(duration) or not 0.05 <= float(duration) <= 30:
                    errors.append(f"{prefix}持续时间必须为 0.05–30 秒")
                    continue
                if direction not in definition["directions"]:
                    errors.append(f"{prefix}方向无效")
                    continue
                start = _round_time(float(start))
                duration = _round_time(float(duration))
                if start + duration > timeline + 1e-6:
                    errors.append(f"{prefix}超出时间轴范围")
                    continue
                parsed_segments.append({
                    "id": str(raw_segment.get("id") or uuid.uuid4().hex[:10]),
                    "start": start,
                    "duration": duration,
                    "direction": direction,
                })
            parsed_segments.sort(key=lambda item: (item["start"], item["id"]))
            previous_end = 0.0
            for segment in parsed_segments:
                if segment["start"] < previous_end - 1e-6:
                    errors.append(f"{definition['label']} 的动作段发生时间重叠")
                previous_end = max(previous_end, segment["start"] + segment["duration"])
                resource_intervals.append({
                    "resource": str(motor),
                    "start": segment["start"],
                    "end": segment["start"] + segment["duration"],
                    "kind": "motor_segment",
                    "label": f"{definition['label']}动作段",
                })
            normalized_motor_tracks.append({"motor": motor, "segments": parsed_segments})

        raw_actions = document.get("actions", [])
        if not isinstance(raw_actions, list):
            errors.append("actions 必须是数组")
            raw_actions = []
        action_catalog = {item["id"]: item for item in self.action_catalog()}
        normalized_actions = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                errors.append(f"已有动作 {index + 1} 格式无效")
                continue
            action_name = raw_action.get("sequence_name")
            start = raw_action.get("start")
            if action_name not in action_catalog:
                errors.append(f"已有动作不存在: {action_name}")
                continue
            if not _finite_number(start) or float(start) < 0:
                errors.append(f"已有动作 {action_name} 的开始时间无效")
                continue
            start = _round_time(float(start))
            if start + action_catalog[action_name]["duration"] > timeline + 1e-6:
                errors.append(f"已有动作 {action_name} 超出时间轴范围")
                continue
            normalized_actions.append({
                "id": str(raw_action.get("id") or uuid.uuid4().hex[:10]),
                "sequence_name": action_name,
                "start": start,
                "remark": str(raw_action.get("remark") or "").strip(),
            })
            for resource in action_catalog[action_name]["channels"]:
                resource_intervals.append({
                    "resource": resource,
                    "start": start,
                    "end": start + action_catalog[action_name]["duration"],
                    "kind": "action",
                    "label": f"已有动作 {action_name}",
                })

        conflict_messages: set[str] = set()
        intervals_by_resource: dict[str, list[dict[str, Any]]] = {}
        for interval in resource_intervals:
            intervals_by_resource.setdefault(interval["resource"], []).append(interval)
        for resource, intervals in intervals_by_resource.items():
            intervals.sort(key=lambda item: (item["start"], item["end"]))
            for index, current in enumerate(intervals):
                for other in intervals[index + 1:]:
                    if other["start"] >= current["end"] - 1e-6:
                        break
                    if current["kind"] == other["kind"] == "segment":
                        continue
                    overlap_start = max(current["start"], other["start"])
                    conflict_messages.add(
                        f"机构 {resource} 在 {overlap_start:g}s 同时被"
                        f"{current['label']}和{other['label']}控制"
                    )
        errors.extend(sorted(conflict_messages))

        raw_audio = document.get("audio")
        normalized_audio = None
        if raw_audio is not None:
            if not isinstance(raw_audio, Mapping):
                errors.append("audio 必须是音乐资源对象")
            else:
                asset_id = raw_audio.get("asset_id")
                duration = raw_audio.get("duration")
                metadata = next(
                    (item for item in self.list_audio() if item.get("id") == asset_id),
                    None,
                )
                if metadata is None:
                    errors.append("选择的音乐资源不存在，请重新上传")
                stored_duration = metadata.get("duration") if metadata is not None else duration
                if not _finite_number(stored_duration) or not 0.1 <= float(stored_duration) <= MAX_TIMELINE_SECONDS:
                    errors.append(
                        f"音乐时长必须为 0.1–{int(MAX_TIMELINE_SECONDS)} 秒"
                    )
                elif float(stored_duration) > timeline + 0.05:
                    errors.append("音乐长度超过时间轴，请先延长时间轴")
                if metadata is not None and _finite_number(stored_duration):
                    normalized_audio = {
                        "asset_id": asset_id,
                        "name": str(metadata.get("name") or asset_id),
                        "duration": _round_time(float(stored_duration)),
                    }

        if errors:
            raise ChoreographyError("动作编排校验失败", errors)

        # Build the compact timeline consumed by the motion node.  Motor
        # tracks remain in the saved document, but are intentionally omitted
        # from this preview payload so a Web preview can never drive the
        # treads.
        preview_by_time: dict[float, list[dict[str, Any]]] = {}
        for transition in compiled_transitions:
            from_targets = transition.get("from_targets")
            to_targets = transition.get("to_targets")
            if not isinstance(from_targets, Mapping) or not isinstance(to_targets, Mapping):
                continue
            duration = max(0.05, float(transition["duration"]))
            max_delta = max(
                (abs(float(to_targets[name]) - float(from_targets.get(name, to_targets[name])))
                 for name in to_targets),
                default=0.0,
            )
            step_size = max(1.0, min(1000.0, max_delta / (duration * 50.0)))
            preview_by_time.setdefault(float(transition["start"]), []).append({
                "type": "manual_servo",
                "targets": dict(to_targets),
                "step_size": round(step_size, 2),
            })
        sequences, poses, _ = self._sequence_catalog_data()
        library = SequenceLibrary(sequences, poses)
        for action in normalized_actions:
            for frame in library.flatten(action["sequence_name"], offset_time=action["start"]):
                safe_actions = [
                    item for item in frame.get("actions", [])
                    if isinstance(item, Mapping) and item.get("type") != "motor"
                ]
                if safe_actions:
                    preview_by_time.setdefault(float(frame["time"]), []).extend(
                        deepcopy(safe_actions)
                    )
        preview_frames = [
            {"time": _round_time(timestamp), "actions": actions}
            for timestamp, actions in sorted(preview_by_time.items())
        ]

        normalized = {
            "schema_version": SCHEMA_VERSION,
            "id": choreography_id,
            "name": name.strip(),
            "remark": remark,
            "start_pose": "neutral",
            "timeline_seconds": _round_time(timeline),
            "tracks": normalized_tracks,
            "motor_tracks": normalized_motor_tracks,
            "actions": sorted(normalized_actions, key=lambda item: item["start"]),
            "audio": normalized_audio,
        }
        compiled = {
            "schema_version": SCHEMA_VERSION,
            "id": choreography_id,
            "name": name.strip(),
            "remark": remark,
            "start_pose": "neutral",
            "duration": _round_time(timeline),
            "transitions": sorted(
                compiled_transitions, key=lambda item: (item["start"], item["channel"])
            ),
            "motor_tracks": deepcopy(normalized["motor_tracks"]),
            "actions": deepcopy(normalized["actions"]),
            "audio": deepcopy(normalized_audio),
            "preview": {
                "duration": _round_time(timeline),
                "frames": preview_frames,
            },
        }
        return normalized, compiled

    @staticmethod
    def _targets_for(
        channel: str, percent: float, servos: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, int]:
        normalized = max(-1.0, min(1.0, float(percent) / 100.0))
        if channel == "neck_pitch":
            return neck_kinematics_from_servos(dict(servos)).targets(normalized)
        servo = servos[channel]
        initial = int(servo["init"])
        low, high = sorted((int(servo["limit_1"]), int(servo["limit_2"])))
        destination = high if normalized >= 0 else low
        return {channel: int(round(initial + (destination - initial) * abs(normalized)))}

