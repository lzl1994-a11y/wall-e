"""Run ROS callbacks and realtime media on one Python scheduling thread."""
import asyncio


async def run_ros_media(node, media, stopping, *, media_active=None):
    from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)

    async def ros_callbacks():
        # A peer needs polling at 1/10 of its 20 ms audio block period.
        # Without a peer, one block is enough for status callbacks; rebuilding
        # an empty ROS wait set every 2 ms otherwise burns CPU while idle.
        # Keep one owner thread so native I/O cannot reintroduce GIL contention.
        while node.context.ok() and not stopping.is_set():
            try:
                executor.spin_once(timeout_sec=0)
            except ExternalShutdownException:
                if node.context.ok():
                    raise
                return
            active = media_active is None or media_active()
            await asyncio.sleep(0.002 if active else 0.020)

    tasks = [asyncio.create_task(ros_callbacks()), asyncio.create_task(media())]
    try:
        completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in completed:
            task.result()
    finally:
        stopping.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        executor.remove_node(node)
        executor.shutdown(timeout_sec=0)
        for result in results:
            if isinstance(result, Exception):
                raise result
