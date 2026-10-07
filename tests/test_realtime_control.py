from services.remote.realtime_control import ChangedTargetGate
from services.remote.realtime_control import ChannelSequenceGate
import pytest


def test_motion_reordering_drops_old_vector_without_rejecting_reliable_commands():
    gate = ChannelSequenceGate()
    assert gate.accept("motion", {"type": "control", "seq": 10})
    assert not gate.accept("motion", {"type": "control", "seq": 9})
    assert gate.accept("control", {"type": "voice", "seq": 1})
    assert gate.accept("motion", {"type": "control", "seq": 11})


def test_reliable_sequence_replay_and_commands_on_motion_are_rejected():
    gate = ChannelSequenceGate()
    assert gate.accept("control", {"type": "voice", "seq": 1})
    with pytest.raises(ValueError):
        gate.accept("control", {"type": "voice", "seq": 1})
    with pytest.raises(ValueError):
        gate.accept("motion", {"type": "action", "seq": 10})


def test_stop_latches_both_channels_until_new_session():
    gate = ChannelSequenceGate()
    assert gate.accept("control", {"type": "stop", "seq": 1})
    assert not gate.accept("motion", {"type": "control", "seq": 999})
    assert not gate.accept("control", {"type": "action", "seq": 2})
    gate.reset()
    assert gate.accept("motion", {"type": "control", "seq": 1})


def test_stationary_target_is_not_reissued_on_every_heartbeat():
    gate = ChangedTargetGate(min_interval_sec=0.1)
    center = {"head_yaw": 5000, "neck": 4200}

    assert gate.accept(center, 1.0)
    assert not gate.accept(center, 1.1)
    assert not gate.accept(center, 10.0)


def test_latest_changed_target_passes_after_rate_limit():
    gate = ChangedTargetGate(min_interval_sec=0.1)

    assert gate.accept({"head_yaw": 5000}, 1.0)
    assert not gate.accept({"head_yaw": 4900}, 1.05)
    assert gate.accept({"head_yaw": 4800}, 1.10)


def test_reset_allows_center_target_for_a_new_connection():
    gate = ChangedTargetGate(min_interval_sec=0.1)
    center = {"head_yaw": 5000}

    assert gate.accept(center, 1.0)
    gate.reset()
    assert gate.accept(center, 1.01)
