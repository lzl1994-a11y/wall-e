"""Run the dependency-free browser state regressions when Node is available."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_eye_frontend_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser state regressions")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_suffix(".cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
