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

        # 1. Non-zero left stick drives differential mixing
        node._axes[module.AXIS_LY] = 0.6
        node._axes[module.AXIS_LX] = 0.2

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 1)
        self.assertEqual(len(node.action_pub.messages), 1)
        motor_cmd = json.loads(node.motor_pub.messages[0].data)
        self.assertIn("left", motor_cmd)
        self.assertIn("right", motor_cmd)
        self.assertTrue(node._was_moving)

        # 2. Continuous motion on subsequent tick continues to publish every tick
        node._tick_loop()
        self.assertEqual(len(node.motor_pub.messages), 2)
        self.assertEqual(len(node.action_pub.messages), 2)
        self.assertTrue(node._was_moving)

        # 3. Resetting axes stops motors once and clears moving flag
        node._axes[module.AXIS_LY] = 0.0
        node._axes[module.AXIS_LX] = 0.0

        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 3)
        self.assertEqual(len(node.action_pub.messages), 3)
        stop_cmd = json.loads(node.motor_pub.messages[2].data)
        self.assertEqual(stop_cmd, module.STOP_COMMAND)
        self.assertFalse(node._was_moving)

        # 4. Continuous idle does NOT repeat stop command, while manual_servo is still published
        node._tick_loop()

        self.assertEqual(len(node.motor_pub.messages), 3)
        self.assertEqual(len(node.action_pub.messages), 4)
        self.assertFalse(node._was_moving)

    def test_failed_stop_publish_keeps_moving_state_for_retry(self):
        module = _load_module()
        node = self._create_node(module)
        node._was_moving = True

        with patch.object(node.motor_pub, "publish", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                node._tick_loop()

        self.assertTrue(node._was_moving)
        node._tick_loop()
        self.assertEqual(len(node.motor_pub.messages), 1)
        self.assertEqual(json.loads(node.motor_pub.messages[0].data), module.STOP_COMMAND)
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

    def test_invalid_axis_range_is_neutral_and_does_not_stop_event_loop(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True
        node._axes[module.AXIS_LY] = 0.6
        node.device.capabilities = lambda verbose=False: {
            3: [
                (module.AXIS_LY, _AbsInfo(100, 100)),
                (module.AXIS_LX, _AbsInfo(-100, 100)),
            ]
        }
        node.device.read_loop = lambda: [
            _Event(3, module.AXIS_LY, 100),
            _Event(3, module.AXIS_LX, 50),
        ]

        node._run_control()

        self.assertEqual(node._axes[module.AXIS_LY], 0.0)
        self.assertEqual(node._axes[module.AXIS_LX], 0.5)

    def test_invalid_drive_axis_range_stops_previous_motion_on_next_tick(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True
        node._was_moving = True
        node._axes[module.AXIS_LY] = 0.6
        node.device.capabilities = lambda verbose=False: {
            3: [(module.AXIS_LY, _AbsInfo(100, 100))]
        }
        node.device.read_loop = lambda: [_Event(3, module.AXIS_LY, 100)]

        node._run_control()
        node._tick_loop()

        self.assertEqual(node._axes[module.AXIS_LY], 0.0)
        self.assertEqual(json.loads(node.motor_pub.messages[-1].data), module.STOP_COMMAND)
        self.assertFalse(node._was_moving)

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

    def test_button_x_and_y_release_publishes_action_command(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        # Press X then release X -> publishes wave_hello
        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_X, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_X, 0),
        ]
        node._run_control()

        self.assertEqual(len(node.action_pub.messages), 1)
        msg_x = json.loads(node.action_pub.messages[0].data)
        self.assertEqual(msg_x["name"], "wave_hello")
        self.assertEqual(msg_x["source"], "joystick")

        # Press Y then release Y -> publishes raise_hand
        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 0),
        ]
        node._run_control()

        self.assertEqual(len(node.action_pub.messages), 2)
        msg_y = json.loads(node.action_pub.messages[1].data)
        self.assertEqual(msg_y["name"], "raise_hand")
        self.assertEqual(msg_y["source"], "joystick")

    def test_button_a_and_b_press_publishes_action_command(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_A, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_B, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_A, 0),  # release ignored
            _Event(module.ecodes.EV_KEY, module.BTN_B, 2),  # repeat ignored
        ]
        node._run_control()

        self.assertEqual(len(node.action_pub.messages), 2)
        msg_a = json.loads(node.action_pub.messages[0].data)
        self.assertEqual(msg_a["name"], "happy_dance")
        msg_b = json.loads(node.action_pub.messages[1].data)
        self.assertEqual(msg_b["name"], "sad_react")

    def test_button_l1_and_r1_press_updates_eyebrow_timers(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        with patch("time.time", return_value=1000.0):
            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_KEY, module.BTN_L1, 1),
                _Event(module.ecodes.EV_KEY, module.BTN_R1, 1),
            ]
            node._run_control()

        self.assertEqual(node._auto_timers["eyebrow_l"], 1003.0)
        self.assertEqual(node._auto_timers["eyebrow_r"], 1003.0)

    def test_hat_directions_update_arm_timers_via_run_control(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        with patch("time.time", return_value=2000.0):
            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_ABS, module.HAT_X, -1),
            ]
            node._run_control()
            self.assertEqual(node._auto_timers["arm_l"], 2003.0)
            self.assertEqual(node._auto_timers["arm_r"], 0.0)

            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_ABS, module.HAT_X, 1),
            ]
            node._run_control()
            self.assertEqual(node._auto_timers["arm_r"], 2003.0)

            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_ABS, module.HAT_Y, 1),
            ]
            node._run_control()
            self.assertEqual(node._auto_timers["arm_l"], 0.0)
            self.assertEqual(node._auto_timers["arm_r"], 0.0)

    def test_chord_hold_2s_triggers_game_toggle_and_suppresses_release_actions(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        # Use mock monotonic clock for chord hold
        now_mono = [100.0]
        node._button_policy = module.JoystickButtonPolicy(
            hold_seconds=2.0,
            chord_clock=lambda: now_mono[0],
        )

        # Press X and Y
        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_X, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 1),
        ]
        node._run_control()

        # Advance clock and tick
        now_mono[0] += 2.0
        node._tick_loop()

        # Game toggle request was published
        self.assertEqual(len(node.game_request_pub.messages), 1)
        req = json.loads(node.game_request_pub.messages[0].data)
        self.assertEqual(req["request"], "toggle")
        self.assertEqual(req["controller"], "/dev/input/event2")
        self.assertTrue(node._button_policy.chord_fired)

        # Releasing X and Y must NOT send wave_hello or raise_hand
        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_X, 0),
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 0),
        ]
        node._run_control()
        self.assertEqual(node.action_pub.messages, [])
        self.assertFalse(node._button_policy.chord_fired)

    def test_game_mode_active_suppresses_buttons_but_keeps_hat(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True

        # Receive game state active message
        node._on_game_state(_String(data=json.dumps({"mode": "menu"})))
        self.assertTrue(node._game_active)

        # Pressing A, B, L1, R1, X, Y produces no action command
        with patch("time.time", return_value=500.0):
            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_KEY, module.BTN_A, 1),
                _Event(module.ecodes.EV_KEY, module.BTN_B, 1),
                _Event(module.ecodes.EV_KEY, module.BTN_L1, 1),
                _Event(module.ecodes.EV_KEY, module.BTN_X, 1),
                _Event(module.ecodes.EV_KEY, module.BTN_X, 0),
            ]
            node._run_control()

        self.assertEqual(node.action_pub.messages, [])
        self.assertEqual(node._auto_timers["eyebrow_l"], 0.0)

        # HAT events are NOT suppressed by game mode
        with patch("time.time", return_value=500.0):
            node.device.read_loop = lambda: [
                _Event(module.ecodes.EV_ABS, module.HAT_X, -1),
            ]
            node._run_control()

        self.assertEqual(node._auto_timers["arm_l"], 503.0)

    def test_game_mode_change_suppresses_pending_chord_toggle(self):
        module = _load_module()
        node = self._create_node(module)
        node.running = True
        now_mono = [100.0]
        node._button_policy = module.JoystickButtonPolicy(
            hold_seconds=2.0,
            chord_clock=lambda: now_mono[0],
        )
        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_X, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 1),
        ]
        node._run_control()

        now_mono[0] += 2.0
        node._on_game_state(_String(data=json.dumps({"mode": "menu"})))
        node._tick_loop()
        self.assertEqual(node.game_request_pub.messages, [])

        node.device.read_loop = lambda: [
            _Event(module.ecodes.EV_KEY, module.BTN_A, 1),
            _Event(module.ecodes.EV_KEY, module.BTN_Y, 0),
            _Event(module.ecodes.EV_KEY, module.BTN_X, 0),
        ]
        node._run_control()
        self.assertEqual(node.action_pub.messages, [])


if __name__ == "__main__":
    unittest.main()
