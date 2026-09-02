#!/usr/bin/env python3
"""Single authority for motor priority, validation, and upstream timeout."""

import json
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from services.motion_arbiter import (
    MOTOR_OUTPUT_TOPIC,
    SOURCE_TOPICS,
    MotionArbiter,
    STOP_COMMAND,
)
from services.game_protocol import GAME_MODE_STATE_TOPIC, game_is_active


# Some deployed Humble/FastDDS combinations return from the executor wait set
# continuously even when no application callback is ready. Yielding briefly
# caps that idle spin without adding meaningful latency to track commands.
EXECUTOR_YIELD_SEC = 0.005


class MotionArbiterNode(Node):
    def __init__(self, *, arbiter=None, start_watchdog=True):
        # This node has no runtime parameters, and console logging is enough.
        # Disabling the default parameter services and /rosout publisher keeps
        # dozens of unused DDS/QoS entities out of the executor wait set.
        super().__init__(
            "motion_arbiter_node",
            enable_rosout=False,
            start_parameter_services=False,
        )
        self._arbiter = arbiter or MotionArbiter()
        self._publisher = self.create_publisher(String, MOTOR_OUTPUT_TOPIC, 10)
        self._last_source = None
        self._last_command = None
        self._game_active = False
        self._operation_lock = threading.RLock()
        self._watchdog_condition = threading.Condition()
        self._watchdog_deadline = None
        self._watchdog_closed = False
        self._watchdog_thread = None
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self._subscriptions = [
            self.create_subscription(
                String,
                topic,
                lambda message, source=source: self._on_command(source, message),
                10,
            )
            for source, topic in SOURCE_TOPICS.items()
        ]
        if start_watchdog:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="motor-arbiter-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()
        self.get_logger().info(
            "运动仲裁器上线：joystick > tracking > autonomy，"
            "事件驱动，上游命令超时自动停车"
        )

    def _on_command(self, source, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning(f"拒绝无效 {source} 电机 JSON")
            return
        with self._operation_lock:
            if self._game_active:
                return
            with self._watchdog_condition:
                if not self._arbiter.update(source, payload):
                    self.get_logger().warning(f"拒绝无效 {source} 电机指令")
                    return
                selected_source, command, deadline = self._arbiter.select_with_deadline()
                self._set_watchdog_deadline_locked(deadline)

            # Refresh the hardware watchdog only for the source that currently
            # owns the tracks. Lower-priority traffic must not extend a
            # higher-priority source's lifetime.
            if selected_source == source or selected_source != self._last_source:
                self._publish(selected_source, command, force=True)

    def _publish_selected(self, *, force=False):
        with self._operation_lock:
            if self._game_active:
                source, command = "game-safety", STOP_COMMAND
            else:
                with self._watchdog_condition:
                    source, command, deadline = self._arbiter.select_with_deadline()
                    self._set_watchdog_deadline_locked(deadline)
            self._publish(source, command, force=force)

    def _publish(self, source, command, *, force=False):
        if not force and source == self._last_source and command == self._last_command:
            return
        self._publisher.publish(
            String(data=json.dumps(command, ensure_ascii=False, separators=(",", ":")))
        )
        if source != self._last_source:
            self.get_logger().info(f"电机控制权切换为: {source}")
        self._last_source = source
        self._last_command = command

    def _set_watchdog_deadline_locked(self, deadline):
        self._watchdog_deadline = deadline
        self._watchdog_condition.notify()

    def _watchdog_loop(self):
        while True:
            with self._watchdog_condition:
                while not self._watchdog_closed:
                    deadline = self._watchdog_deadline
                    if deadline is None:
                        self._watchdog_condition.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._watchdog_condition.wait(timeout=remaining)
                        continue
                    self._watchdog_deadline = None
                    break
                if self._watchdog_closed:
                    return
            # rclpy Publisher.publish() is thread-safe. Publishing directly
            # avoids a Humble/FastDDS guard-condition bug observed on the robot
            # where one trigger leaves the executor permanently ready/spinning.
            self._publish_selected()

    def _on_game_state(self, message):
        active = game_is_active(message.data)
        with self._operation_lock:
            was_active = self._game_active
            if active and not was_active:
                with self._watchdog_condition:
                    self._set_watchdog_deadline_locked(None)
                self._publish("game-safety", STOP_COMMAND, force=True)
            self._game_active = active
            if was_active and not active:
                self._publish_selected(force=True)

    def destroy_node(self):
        with self._watchdog_condition:
            self._watchdog_closed = True
            self._watchdog_condition.notify()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
        try:
            self._publish("shutdown", STOP_COMMAND, force=True)
        except Exception:
            # The ROS context may already be invalid during abnormal shutdown.
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionArbiterNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            time.sleep(EXECUTOR_YIELD_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
