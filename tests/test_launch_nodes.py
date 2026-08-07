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


def launcher_args(*, no_web=False):
    return Namespace(
        voice_chat=False,
        real_stt=False,
        keyboard_stt=False,
        no_serial=True,
        tracking=False,
        no_doa=False,
        no_hardware=False,
        no_web=no_web,
    )


class LaunchNodesTests(unittest.TestCase):
    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_config_web_follows_main_launcher(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args())
        web_entries = [entry for entry in entries if entry.name == "config_web"]

        self.assertEqual(len(web_entries), 1)
        self.assertEqual(web_entries[0].script, launch_nodes.ROOT / "services" / "web_server.py")

    @patch("launch_nodes.load_config", return_value={"pipeline": {"mode": "asr_llm"}})
    def test_config_web_can_be_disabled(self, _load_config):
        entries = launch_nodes.build_node_list(launcher_args(no_web=True))

        self.assertNotIn("config_web", [entry.name for entry in entries])

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


if __name__ == "__main__":
    unittest.main()
