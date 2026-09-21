from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_official_behaviortree_ros2_is_pinned_as_a_submodule():
    modules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert "BehaviorTree/BehaviorTree.ROS2.git" in modules
    assert "branch = humble" in modules


def test_bridge_uses_official_execute_tree_interface_and_action_lifecycle():
    source = (
        ROOT
        / "cpp_nodes"
        / "wali_bt_ros2_bridge"
        / "src"
        / "wali_bt_ros2_bridge.cpp"
    ).read_text(encoding="utf-8")
    assert "btcpp_ros2_interfaces::action::ExecuteTree" in source
    assert "rclcpp_action::create_server<ExecuteTree>" in source
    assert 'constexpr char kTreeName[] = "WaliTask"' in source
    assert "handle_goal" in source
    assert "publish_feedback" in source
    assert "handle_cancel" in source
    assert "finish_goal" in source
    assert 'constexpr char kLegacyExecuteTopic[] = "/behavior_tree/execute"' in source


def test_bridge_recovers_a_goal_when_the_legacy_backend_stops_replying():
    source = (
        ROOT
        / "cpp_nodes"
        / "wali_bt_ros2_bridge"
        / "src"
        / "wali_bt_ros2_bridge.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter<double>("goal_timeout_sec", 240.0)' in source
    assert 'declare_parameter<double>("cancel_grace_sec", 2.0)' in source
    assert 'declare_parameter<double>("backend_loss_grace_sec", 2.0)' in source
    assert 'reason = "legacy_behavior_tree_disconnected"' in source
    assert 'reason = "legacy_behavior_tree_cancel_timeout"' in source
    assert 'reason = "legacy_behavior_tree_goal_timeout"' in source
    assert '{"name", "stop_all"}' in source
    assert '{"source", "wali_task_action_bridge_safety"}' in source
    assert "plan_id_ != expected_plan_id" in source
    assert "recovery_in_progress_ = true" in source


def test_bridge_build_keeps_multiarch_workaround_outside_upstream_submodule():
    build_script = (ROOT / "tools" / "build_behaviortree_ros2.sh").read_text(
        encoding="utf-8"
    )
    assert "wali_bt_ros2_bridge" in build_script
    assert "-Dbehaviortree_cpp_DIR=" in build_script
    config = (
        ROOT / "cmake" / "behaviortree_cpp" / "behaviortree_cppConfig.cmake"
    ).read_text(encoding="utf-8")
    assert "lib/aarch64-linux-gnu" in config
    assert "behaviortree_cpp::behaviortree_cpp" in config
