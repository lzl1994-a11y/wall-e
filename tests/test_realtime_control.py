from services.remote.realtime_control import ChangedTargetGate


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
