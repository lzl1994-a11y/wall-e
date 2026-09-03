import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import launch_nodes


def launcher_args(
    *,
    no_web=False,
    no_serial=True,
    no_hardware=False,
    tracking=False,
    keyboard_stt=False,
    mcp=False,
    no_mcp=False,
    save_voice_debug=False,
):
    return Namespace(
        voice_chat=False,
        real_stt=False,
        keyboard_stt=keyboard_stt,
        no_serial=no_serial,
        tracking=tracking,
        no_doa=False,
        no_hardware=no_hardware,
        no_web=no_web,
        mcp=mcp,
        no_mcp=no_mcp,
        save_voice_debug=save_voice_debug,
    )


class LaunchNodesTests(unittest.TestCase):
    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_keyboard_flag_selects_existing_keyboard_node(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(keyboard_stt=True))
        keyboard = next(entry for entry in entries if entry.name == "keyboard_stt")

        self.assertTrue(keyboard.script.is_file())
        self.assertNotIn("stt", [entry.name for entry in entries])

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
            "hardware": {"backend": "serial_mcu"},
        },
    )
    def test_serial_mcu_backend_starts_serial_bridge(self, _load_config):
        names = [
            entry.name
            for entry in launch_nodes.build_node_list(launcher_args(no_serial=False))
        ]
        self.assertIn("serial", names)
        self.assertIn("hardware_bridge", names)
        self.assertLess(names.index("motion_arbiter"), names.index("hardware_bridge"))
        self.assertNotIn("i2c_hardware", names)

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
        },
    )
    def test_dialog_motion_starts_after_action_owner(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(no_serial=False))
        names = [entry.name for entry in entries]
        dialog_motion = next(entry for entry in entries if entry.name == "dialog_motion")

        self.assertTrue(dialog_motion.script.is_file())
        self.assertLess(names.index("action"), names.index("dialog_motion"))

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
            "hardware": {"backend": "ubuntu_i2c"},
        },
    )
    def test_ubuntu_i2c_backend_starts_single_i2c_owner(self, _load_config):
        names = [
            entry.name
            for entry in launch_nodes.build_node_list(launcher_args(no_serial=False))
        ]
        self.assertIn("serial", names)
        self.assertIn("i2c_hardware", names)
        self.assertLess(names.index("motion_arbiter"), names.index("i2c_hardware"))
        self.assertNotIn("hardware_bridge", names)
        self.assertNotIn("servo", names)
        self.assertNotIn("motor", names)

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
            "hardware": {"backend": "ubuntu_i2c"},
        },
    )
    def test_no_hardware_skips_selected_backend(self, _load_config):
        names = [
            entry.name
            for entry in launch_nodes.build_node_list(
                launcher_args(no_serial=False, no_hardware=True)
            )
        ]
        self.assertNotIn("i2c_hardware", names)
        self.assertNotIn("hardware_bridge", names)

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_config_web_follows_main_launcher(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args())
        web_entries = [entry for entry in entries if entry.name == "config_web"]

        self.assertEqual(len(web_entries), 1)
        self.assertEqual(web_entries[0].script, launch_nodes.ROOT / "services" / "web_server.py")

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_camera_capture_owner_always_starts_before_consumers(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args())
        names = [entry.name for entry in entries]

        self.assertEqual(names[0], "camera_capture")
        self.assertLess(names.index("camera_capture"), names.index("config_web"))
        self.assertLess(names.index("camera_capture"), names.index("llm"))
        self.assertEqual(
            entries[0].script,
            launch_nodes.ROOT / "nodes" / "camera_capture_node.py",
        )

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_tft_tcp_service_has_one_independent_owner(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args())
        tft_entries = [entry for entry in entries if entry.name == "tft_tcp_service"]
        names = [entry.name for entry in entries]

        self.assertEqual(len(tft_entries), 1)
        self.assertEqual(
            tft_entries[0].script,
            launch_nodes.ROOT / "nodes" / "tft_tcp_service_node.py",
        )
        self.assertLess(names.index("camera_capture"), names.index("tft_tcp_service"))
        self.assertLess(names.index("tft_tcp_service"), names.index("llm"))

    def test_dialogue_nodes_do_not_own_tft_tcp_server(self):
        for filename in ("llm_ros_node.py", "voice_chat_ros_node.py"):
            source = (launch_nodes.ROOT / "nodes" / filename).read_text(encoding="utf-8")
            self.assertNotIn("GameTftStreamServer", source)
            self.assertNotIn("TrackingTftPreview", source)
            self.assertNotIn("tft_preview_ready", source)

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_config_web_can_be_disabled(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(no_web=True))

        self.assertNotIn("config_web", [entry.name for entry in entries])

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
            "mcp": {"enabled": False},
        },
    )
    def test_mcp_flag_adds_gateway_after_action_owner(self, _load_config):
        names = [
            entry.name
            for entry in launch_nodes.build_node_list(launcher_args(mcp=True))
        ]
        self.assertIn("mcp_gateway", names)
        self.assertLess(names.index("action"), names.index("mcp_gateway"))

    @patch(
        "launch_nodes.load_config",
        return_value={
            "pipeline": {"mode": "asr_llm"},
            "launch": {"serial": True, "tracking": False},
            "mcp": {"enabled": True},
        },
    )
    def test_no_mcp_flag_overrides_enabled_config(self, _load_config):
        names = [
            entry.name
            for entry in launch_nodes.build_node_list(launcher_args(no_mcp=True))
        ]
        self.assertNotIn("mcp_gateway", names)

    @patch("launch_nodes.subprocess.Popen")
    def test_child_process_can_import_project_packages(self, popen):
        entry = launch_nodes.NodeEntry("test", launch_nodes.ROOT / "launch_nodes.py")

        with patch.dict(os.environ, {"PYTHONPATH": "existing-path"}):
            launch_nodes.start_process(entry)

        child_env = popen.call_args.kwargs["env"]
        self.assertEqual(
            child_env["PYTHONPATH"].split(os.pathsep),
            [str(launch_nodes.ROOT), "existing-path"],
        )

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_voice_debug_flag_is_scoped_to_voice_pipeline_nodes(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(save_voice_debug=True))
        by_name = {entry.name: entry for entry in entries}

        self.assertEqual(by_name["stt"].environment["WALI_SAVE_VOICE_DEBUG"], "1")
        self.assertEqual(by_name["llm"].environment["WALI_SAVE_VOICE_DEBUG"], "1")
        self.assertEqual(by_name["camera_capture"].environment, {})

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_voice_debug_is_explicitly_disabled_without_flag(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args())
        by_name = {entry.name: entry for entry in entries}

        self.assertEqual(by_name["stt"].environment["WALI_SAVE_VOICE_DEBUG"], "0")
        self.assertEqual(by_name["llm"].environment["WALI_SAVE_VOICE_DEBUG"], "0")

    @patch("launch_nodes.subprocess.Popen")
    def test_node_environment_overrides_inherited_debug_setting(self, popen):
        entry = launch_nodes.NodeEntry(
            "test",
            launch_nodes.ROOT / "launch_nodes.py",
            environment={"WALI_SAVE_VOICE_DEBUG": "0"},
        )

        with patch.dict(os.environ, {"WALI_SAVE_VOICE_DEBUG": "1"}):
            launch_nodes.start_process(entry)

        self.assertEqual(
            popen.call_args.kwargs["env"]["WALI_SAVE_VOICE_DEBUG"],
            "0",
        )

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_tracking_node_loads_tros_environment(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(tracking=True))
        tracking = next(entry for entry in entries if entry.name == "tracking")

        self.assertEqual(
            tracking.environment_setup,
            Path("/opt/tros/humble/setup.bash"),
        )

    @patch("launch_nodes.subprocess.Popen")
    def test_environment_setup_wraps_node_process(self, popen):
        setup = Path("/opt/tros/humble/setup.bash")
        entry = launch_nodes.NodeEntry(
            "tracking",
            launch_nodes.ROOT / "launch_nodes.py",
            environment_setup=setup,
        )

        with patch.object(launch_nodes.os, "name", "posix"):
            launch_nodes.start_process(entry)

        command = popen.call_args.args[0]
        self.assertEqual(command[:3], ["bash", "-c", 'source "$1" && exec "$2" "$3"'])
        self.assertEqual(command[4], str(setup))


if __name__ == "__main__":
    unittest.main()
