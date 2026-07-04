#!/usr/bin/env python3
"""Deprecated: the NV12 padding pipeline was removed because it capped vision at ~1 FPS."""

import sys


if __name__ == "__main__":
    print(
        "nv12_padder_node.py is deprecated. "
        "Use hobot_vision_node.py direct /image_nv12 + ai_msg_scaler_node.py instead.",
        file=sys.stderr,
    )
    sys.exit(1)
