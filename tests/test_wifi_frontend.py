"""Run Wi-Fi page state regressions without a browser dependency."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_wifi_frontend_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend regressions")
    result = subprocess.run([node, "--test", str(Path(__file__).with_suffix(".cjs"))],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
