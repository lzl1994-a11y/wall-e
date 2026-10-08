import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("rclpy")
from services.remote.ros_media_loop import run_ros_media


@pytest.mark.parametrize("fail", [False, True])
def test_ros_callbacks_share_media_thread_and_failures_close_both_owners(fail):
    events = []
    owner = threading.get_ident()
    stopped = threading.Event()
    node = SimpleNamespace(context=SimpleNamespace(ok=lambda: True))

    class Executor:
        def __init__(self, *, context):
            assert context is node.context

        def add_node(self, value):
            assert value is node

        def spin_once(self, *, timeout_sec):
            assert timeout_sec == 0
            assert threading.get_ident() == owner
            events.append("ros")
            if fail:
                raise RuntimeError("ROS callback failed")

        def remove_node(self, value):
            events.append("remove")

        def shutdown(self, *, timeout_sec):
            events.append("shutdown")

    async def media():
        assert threading.get_ident() == owner
        try:
            for _ in range(5):
                events.append("media")
                await asyncio.sleep(.002)
        finally:
            events.append("closed")

    with patch("rclpy.executors.SingleThreadedExecutor", Executor):
        if fail:
            with pytest.raises(RuntimeError, match="ROS callback failed"):
                asyncio.run(run_ros_media(node, media, stopped))
        else:
            asyncio.run(run_ros_media(node, media, stopped))
    assert "ros" in events and "media" in events and "closed" in events
    assert events[-2:] == ["remove", "shutdown"]
    assert stopped.is_set()
