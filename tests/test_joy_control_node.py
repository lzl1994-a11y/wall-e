import importlib
import json
import sys
import types
import unittest
from unittest.mock import patch


class _String:
    def __init__(self, data=""):
        self.data = data


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Logger:
    def info(self, _message):
        pass


class _Node:
    def __init__(self, _name):
        self.publishers = {}
        self.subscriptions = {}
        self.timers = []

    def create_publisher(self, _message_type, topic, _qos):
        publisher = _Publisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, _message_type, topic, callback, _qos):
        self.subscriptions[topic] = callback
        return object()

    def create_timer(self, interval, callback):
        timer = (interval, callback)
        self.timers.append(timer)
        return timer

    def destroy_timer(self, timer):
        if timer in self.timers:
            self.timers.remove(timer)

    def get_logger(self):
        return _Logger()


class _Event:
    def __init__(self, type_, code, value):
        self.type = type_
        self.code = code
        self.value = value


class _AbsInfo:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val


def _load_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _Node
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _String

    fake_evdev = types.ModuleType("evdev")
    fake_ecodes = types.ModuleType("evdev.ecodes")
    fake_ecodes.EV_ABS = 3
    fake_ecodes.EV_KEY = 1
    fake_evdev.ecodes = fake_ecodes
    fake_evdev.list_devices = lambda: []

    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "std_msgs.msg": fake_std,
        "evdev": fake_evdev,
        "evdev.ecodes": fake_ecodes,
    }
    sys.modules.pop("nodes.joy_control_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.joy_control_node")


class JoyControlNodeContractTests(unittest.TestCase):
    def _create_node(self, module):
        with patch.object(module.JoyControlNode, "_start_scanning"):
            node = module.JoyControlNode()
        node.device = types.SimpleNamespace(path="/dev/input/event2", name="FakeJoy")
        return node

    def test_tick_publishes_manual_servo_payload(self):
        module = _load_module()
        node = self._create_node(module)

        # Set axes for right stick and triggers
        node._axes[module.AXIS_RX] = 0.5
        node._axes[module.AXIS_RY] = 0.0
        node._axes[module.AXIS_L2] = 0.4
        node._axes[module.AXIS_R2] = 0.6
        node._auto_timers["arm_l"] = 0.0
        node._auto_timers["arm_r"] = 0.0
        node._auto_timers["eyebrow_l"] = 0.0
        node._auto_timers["eyebrow_r"] = 0.0

        node._tick_loop()

        self.assertEqual(len(node.action_pub.messages), 1)
        payload = json.loads(node.action_pub.messages[0].data)

        self.assertEqual(payload["name"], "manual_servo")
        self.assertEqual(payload["source"], "joystick")
        self.assertEqual(payload["arguments"]["step_size"], node.servo_step_size)

        targets = payload["arguments"]["targets"]
        self.assertEqual(targets["head_yaw"], int(5000 - 0.5 * 2600))
        self.assertEqual(targets["eye_l"], int(7500 - 0.4 * 2500))
        self.assertEqual(targets["eye_r"], int(2000 + 0.6 * 2000))
        self.assertEqual(targets["arm_l"], 2000)
        self.assertEqual(targets["arm_r"], 8000)
        self.assertEqual(targets["eyebrow_l"], 8000)
        self.assertEqual(targets["eyebrow_r"], 1920)
        self.assertIn("neck_top", targets)
        self.assertIn("neck_bottom", targets)

        # Verify exact target keys order
        self.assertEqual(
            list(targets.keys()),
            [
                "head_yaw",
                "neck_top",
                "neck_bottom",
                "eye_l",
                "eye_r",
                "arm_l",
                "arm_r",
                "eyebrow_l",
                "eyebrow_r",
            ],
        )

    def test_motor_path_preserved_during_tick(self):
        module = _load_module()
        node = self._create_node(module)

        # Non-zero left stick drives differential mixing
        node._axes[module.AXIS_LY] = 0.6
        node._axes[module.AXIS_LX] = 0.2

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 1)
        motor_cmd = json.loads(node.motor_pub.messages[0].data)
        self.assertIn("left", motor_cmd)
        self.assertIn("right", motor_cmd)
        self.assertTrue(node._was_moving)

        # Resetting axes stops motors
        node._axes[module.AXIS_LY] = 0.0
        node._axes[module.AXIS_LX] = 0.0

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 2)
        stop_cmd = json.loads(node.motor_pub.messages[1].data)
        self.assertEqual(stop_cmd, module.STOP_COMMAND)
        self.assertFalse(node._was_moving)

    def test_game_mode_suppresses_servo_and_motor_output(self):
        module = _load_module()
        node = self._create_node(module)
        node._game_active = True

        node._axes[module.AXIS_RX] = 0.5
        node._axes[module.AXIS_LY] = 0.5

        node._tick_loop()

        self.assertEqual(node.action_pub.messages, [])
        self.assertEqual(node.motor_pub.messages, [])

    def test_ev_abs_stick_event_updates_axes(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        node.device.capabilities = lambda verbose=False: {
            3: [
                (module.AXIS_LX, _AbsInfo(-100, 100)),
                (module.AXIS_LY, _AbsInfo(-100, 100)),
            ]
        }
        # LX=50 -> 0.5 (above deadzone 0.15)
        # LY=50 -> -0.5 (Y-inverted)
        events = [
            _Event(3, module.AXIS_LX, 50),
            _Event(3, module.AXIS_LY, 50),
        ]
        node.device.read_loop = lambda: events

        node._run_control()

        self.assertAlmostEqual(node._axes[module.AXIS_LX], 0.5)
        self.assertAlmostEqual(node._axes[module.AXIS_LY], -0.5)

    def test_ev_abs_trigger_event_updates_axes(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        node.device.capabilities = lambda verbose=False: {
            3: [
                (module.AXIS_L2, _AbsInfo(0, 255)),
                (module.AXIS_R2, _AbsInfo(100, 300)),
            ]
        }
        events = [
            _Event(3, module.AXIS_L2, 127.5),
            _Event(3, module.AXIS_R2, 200),
        ]
        node.device.read_loop = lambda: events

        node._run_control()

        self.assertAlmostEqual(node._axes[module.AXIS_L2], 0.5)
        self.assertAlmostEqual(node._axes[module.AXIS_R2], 0.5)

    def test_missing_capability_info_preserves_axis_value(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        node._axes[module.AXIS_RX] = 0.75
        node.device.capabilities = lambda verbose=False: {3: []}

        events = [_Event(3, module.AXIS_RX, 100)]
        node.device.read_loop = lambda: events

        node._run_control()

        self.assertEqual(node._axes[module.AXIS_RX], 0.75)

    def test_hat_events_do_not_update_axes(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        initial_axes = dict(node._axes)
        events = [
            _Event(3, module.HAT_X, -1),
            _Event(3, module.HAT_Y, 1),
        ]
        node.device.read_loop = lambda: events

        node._run_control()

        self.assertEqual(node._axes, initial_axes)
        # HAT_Y==1 resets timers to 0.0
        self.assertEqual(node._auto_timers["arm_l"], 0.0)
        self.assertEqual(node._auto_timers["arm_r"], 0.0)

    def test_scan_thread_not_started_when_mocked(self):
        module = _load_module()
        node = self._create_node(module)
        self.assertIsNone(node._scan_thread)
        self.assertEqual(node.device.path, "/dev/input/event2")


if __name__ == "__main__":
    unittest.main()
