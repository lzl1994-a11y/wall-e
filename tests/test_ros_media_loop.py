import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("rclpy")
from services.remote.ros_media_loop import run_ros_media


def test_idle_polling_slows_down_and_peer_lifecycle_restores_audio_cadence():
    state = dict(polls=0, active=False)
    delays = []
    stopped = threading.Event()
    node = SimpleNamespace(context=SimpleNamespace(ok=lambda: state["polls"] < 3))
    real_sleep = asyncio.sleep

    class Executor:
        def __init__(self, *, context):
            pass

        def add_node(self, value):
            pass

        def spin_once(self, *, timeout_sec):
            assert timeout_sec == 0
            state["polls"] += 1
            state["active"] = state["polls"] == 2

        def remove_node(self, value):
            pass

        def shutdown(self, *, timeout_sec):
            pass

    async def sleep(delay):
        delays.append(delay)
        await real_sleep(0)

    async def media():
        await asyncio.Event().wait()

    with patch("rclpy.executors.SingleThreadedExecutor", Executor), patch(
        "services.remote.ros_media_loop.asyncio.sleep", sleep
    ):
        asyncio.run(run_ros_media(node, media, stopped,
                                  media_active=lambda: state["active"]))
    assert delays == [.020, .002, .020]
    assert stopped.is_set()


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
