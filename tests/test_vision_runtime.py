import os
import tempfile
import unittest
from pathlib import Path

from services.vision_runtime import VisionArtifactError, require_nv12_padder


class VisionRuntimeArtifactTests(unittest.TestCase):
    def test_missing_binary_fails_without_building(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(VisionArtifactError, "stop the robot"):
                require_nv12_padder(Path(directory), {})

    def test_stale_binary_requires_offline_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "build" / "wali_nv12_padder" / "nv12_padder_node"
            source = root / "cpp_nodes" / "wali_nv12_padder" / "src" / "nv12_padder_node.cpp"
            binary.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            binary.write_bytes(b"binary")
            source.write_text("source", encoding="utf-8")
            os.utime(binary, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))

            with self.assertRaisesRegex(VisionArtifactError, "stale"):
                require_nv12_padder(root, {})

    def test_configured_prebuilt_binary_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "custom-padder"
            binary.write_bytes(b"binary")

            self.assertEqual(
                require_nv12_padder(
                    Path(directory), {"WALI_NV12_PADDER": str(binary)}
                ),
                binary,
            )


if __name__ == "__main__":
    unittest.main()
