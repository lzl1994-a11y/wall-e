from pathlib import Path

import pytest
import yaml

from services.motion.choreography import ChoreographyError, ChoreographyStore


def _config():
    return {
        "servos": [
            {"id": 0, "name": "arm_r", "limit_1": 2000, "limit_2": 8000, "init": 8000},
            {"id": 1, "name": "head_yaw", "limit_1": 1920, "limit_2": 7600, "init": 5000},
            {"id": 2, "name": "neck_top", "limit_1": 2200, "limit_2": 7800, "init": 4000},
            {"id": 3, "name": "neck_bottom", "limit_1": 1800, "limit_2": 7600, "init": 4200},
        ]
    }


def _store(tmp_path: Path) -> ChoreographyStore:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "sequences.yaml").write_text(
        yaml.safe_dump({
            "poses": {
                "right_hand_up": {
                    "default_step": 60,
                    "targets": {"arm_r": 2000},
                }
            },
            "sequences": {
                "wave": [
                    {"time": 0.0, "actions": [{"type": "pose", "name": "right_hand_up"}]},
                    {"time": 0.5, "actions": [{"type": "servo", "name": "arm_r", "pwm": 8000}]},
                ]
            },
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ChoreographyStore(config_path)


def _document():
    return {
        "schema_version": 1,
        "id": "hello_demo",
        "name": "打招呼演示",
        "timeline_seconds": 5,
        "start_pose": "neutral",
        "tracks": [{
            "channel": "head_yaw",
            "segments": [
                {"id": "left", "start": 0.5, "duration": 1.0, "delta": 50},
                {"id": "return", "start": 2.0, "duration": 0.5, "delta": -50},
            ],
        }],
        "actions": [{"id": "wave-1", "sequence_name": "wave", "start": 3.0}],
    }


def test_relative_segments_compile_to_absolute_pwm_targets(tmp_path: Path):
    store = _store(tmp_path)
    normalized, compiled = store.validate(_document())

    assert normalized["start_pose"] == "neutral"
    assert compiled["transitions"][0]["from_targets"] == {"head_yaw": 5000}
    assert compiled["transitions"][0]["to_targets"] == {"head_yaw": 6300}
    assert compiled["transitions"][1]["to_targets"] == {"head_yaw": 5000}
    assert compiled["actions"][0]["sequence_name"] == "wave"


def test_neck_pitch_is_exposed_as_one_logical_channel(tmp_path: Path):
    catalog = _store(tmp_path).catalog()
    neck = next(item for item in catalog["channels"] if item["id"] == "neck_pitch")

    assert neck["members"] == ["neck_top", "neck_bottom"]
    assert next(item for item in catalog["actions"] if item["id"] == "wave")["channels"] == ["arm_r"]


@pytest.mark.parametrize(
    "segments, expected",
    [
        (
            [
                {"start": 0, "duration": 1, "delta": 20},
                {"start": 0.5, "duration": 1, "delta": -20},
            ],
            "时间重叠",
        ),
        (
            [
                {"start": 0, "duration": 1, "delta": 80},
                {"start": 1, "duration": 1, "delta": 30},
            ],
            "超出 -100%～100%",
        ),
    ],
)
def test_invalid_motion_is_rejected(tmp_path: Path, segments, expected):
    store = _store(tmp_path)
    document = _document()
    document["tracks"][0]["segments"] = segments

    with pytest.raises(ChoreographyError) as raised:
        store.validate(document)

    assert any(expected in detail for detail in raised.value.details)


def test_save_list_load_and_delete_are_atomic_from_api_perspective(tmp_path: Path):
    store = _store(tmp_path)
    saved = store.save(_document())

    assert saved["document"]["id"] == "hello_demo"
    assert store.list()[0]["name"] == "打招呼演示"
    assert store.get("hello_demo")["tracks"][0]["channel"] == "head_yaw"

    store.delete("hello_demo")
    assert store.list() == []


def test_audio_can_be_uploaded_listed_read_and_used_by_choreography(tmp_path: Path):
    store = _store(tmp_path)
    audio_bytes = b"RIFF" + b"\x00" * 40

    asset = store.upload_audio(
        "robot-theme.wav", audio_bytes, content_type="audio/wav", duration=4.5
    )

    assert asset["name"] == "robot-theme.wav"
    assert store.list_audio() == [asset]
    assert store.get_audio(asset["id"]) == (audio_bytes, "audio/wav")

    document = _document()
    document["audio"] = {
        "asset_id": asset["id"],
        "name": "ignored-client-name.wav",
        "duration": 4.5,
    }
    normalized, compiled = store.validate(document)

    assert normalized["audio"] == {
        "asset_id": asset["id"],
        "name": "robot-theme.wav",
        "duration": 4.5,
    }
    assert compiled["audio"] == normalized["audio"]


def test_audio_upload_rejects_music_longer_than_timeline_limit(tmp_path: Path):
    store = _store(tmp_path)

    with pytest.raises(ChoreographyError, match="音乐时长"):
        store.upload_audio("too-long.wav", b"RIFF", duration=600.1)


def test_existing_action_cannot_overlap_manual_motion_on_same_mechanism(tmp_path: Path):
    store = _store(tmp_path)
    document = _document()
    document["tracks"] = [{
        "channel": "arm_r",
        "segments": [{"id": "manual", "start": 2.8, "duration": 1, "delta": -20}],
    }]

    with pytest.raises(ChoreographyError) as raised:
        store.validate(document)

    assert any(
        "机构 arm_r" in detail and "已有动作 wave" in detail
        for detail in raised.value.details
    )


def test_existing_actions_cannot_overlap_on_same_mechanism(tmp_path: Path):
    store = _store(tmp_path)
    document = _document()
    document["tracks"] = []
    document["actions"] = [
        {"id": "wave-1", "sequence_name": "wave", "start": 1},
        {"id": "wave-2", "sequence_name": "wave", "start": 1.25},
    ]

    with pytest.raises(ChoreographyError) as raised:
        store.validate(document)

    assert any("机构 arm_r" in detail for detail in raised.value.details)


def test_audio_must_exist_and_fit_inside_timeline(tmp_path: Path):
    store = _store(tmp_path)
    document = _document()
    document["audio"] = {
        "asset_id": "missing12345678.mp3",
        "duration": 7,
    }

    with pytest.raises(ChoreographyError) as raised:
        store.validate(document)

    assert any("选择的音乐资源不存在" in detail for detail in raised.value.details)
    assert any("音乐长度超过时间轴" in detail for detail in raised.value.details)

