"""Validate native vision artifacts without compiling during robot startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class VisionArtifactError(RuntimeError):
    pass


def require_nv12_padder(
    root_dir: Path,
    environment: Mapping[str, str] | None = None,
) -> Path:
    environment = environment or os.environ
    configured = str(environment.get("WALI_NV12_PADDER", "")).strip()
    if configured:
        binary = Path(configured)
        if not binary.is_file():
            raise VisionArtifactError(f"configured NV12 padder not found: {binary}")
        return binary

    binary = root_dir / "build" / "wali_nv12_padder" / "nv12_padder_node"
    sources = (
        root_dir / "cpp_nodes" / "wali_nv12_padder" / "CMakeLists.txt",
        root_dir / "cpp_nodes" / "wali_nv12_padder" / "src" / "nv12_padder_node.cpp",
    )
    if not binary.is_file():
        raise VisionArtifactError(
            "NV12 padder is not built; stop the robot and run "
            "tools/build_nv12_padder.sh before launch"
        )
    try:
        binary_time = binary.stat().st_mtime_ns
        stale = any(source.is_file() and source.stat().st_mtime_ns > binary_time for source in sources)
    except OSError as exc:
        raise VisionArtifactError(f"cannot inspect NV12 padder: {exc}") from exc
    if stale:
        raise VisionArtifactError(
            "NV12 padder is stale; stop the robot and run "
            "tools/build_nv12_padder.sh before launch"
        )
    return binary


__all__ = ["VisionArtifactError", "require_nv12_padder"]
