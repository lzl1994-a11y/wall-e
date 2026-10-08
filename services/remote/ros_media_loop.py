"""Run ROS callbacks and realtime media on one Python scheduling thread."""
import asyncio


async def run_ros_media(node, media, stopping):
    from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)

    async def ros_callbacks():
        # Poll a nonblocking ROS wait set at 1/10 of the audio block period.
        # No ROS thread competes with media for the GIL after native I/O.
        while node.context.ok() and not stopping.is_set():
            try:
                executor.spin_once(timeout_sec=0)
            except ExternalShutdownException:
                if node.context.ok():
                    raise
                return
            await asyncio.sleep(0.002)

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
