import importlib
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, _message):
        self.messages.append(("info", _message))

    def warning(self, _message):
        self.messages.append(("warning", _message))

    def warn(self, _message):
        self.messages.append(("warning", _message))

    def error(self, _message):
        self.messages.append(("error", _message))


class _FakeNode:
    def __init__(self, _name):
        self.publishers = {}
        self.subscription_qos = {}
        self.logger = _FakeLogger()

    def create_publisher(self, _message_type, topic, _qos):
        publisher = _FakePublisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, _message_type, topic, _callback, qos):
        self.subscription_qos[topic] = qos
        return object()

    def create_timer(self, _period, _callback):
        return object()

    def get_logger(self):
        return self.logger

    def destroy_node(self):
        pass


class _FakeQoSProfile:
    def __init__(self, depth):
        self.depth = depth
        self.durability = None


def _load_tracking_module():
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda args=None: None
    fake_rclpy.spin = lambda _node: None
    fake_rclpy.shutdown = lambda: None
    fake_node = types.ModuleType("rclpy.node")
    fake_node.Node = _FakeNode
    fake_qos = types.ModuleType("rclpy.qos")
    fake_qos.QoSProfile = _FakeQoSProfile
    fake_qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient")
    fake_qos.qos_profile_sensor_data = object()
    fake_signals = types.ModuleType("rclpy.signals")
    fake_signals.SignalHandlerOptions = types.SimpleNamespace(NO="no")
    fake_std = types.ModuleType("std_msgs.msg")
    fake_std.String = _FakeString
    fake_std.Int32 = type("Int32", (), {})
    fake_ai = types.ModuleType("ai_msgs.msg")
    fake_ai.PerceptionTargets = type("PerceptionTargets", (), {})
    modules = {
        "rclpy": fake_rclpy,
        "rclpy.node": fake_node,
        "rclpy.qos": fake_qos,
        "rclpy.signals": fake_signals,
        "std_msgs.msg": fake_std,
        "ai_msgs.msg": fake_ai,
    }
    sys.modules.pop("nodes.wali_tracking_node", None)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("nodes.wali_tracking_node")


class WaliTrackingNodeTests(unittest.TestCase):
    def test_stop_all_returns_tracking_to_idle(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        node._set_tracking_mode("follow_me")

        node._on_action_cmd(_FakeString(json.dumps({
            "request_id": "stop",
            "name": "stop_all",
            "arguments": {},
            "source": "safety",
        })))

        self.assertEqual(node.mode, node.MODE_IDLE)
        status = json.loads(node.publishers["/action_status"].messages[-1].data)
        self.assertEqual(status["request_id"], "stop")
        self.assertEqual(status["status"], "completed")

    def test_detection_input_uses_sensor_qos_and_drives_head_target(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        node._set_tracking_mode("look_at_me")
        rect = types.SimpleNamespace(
            x_offset=650,
            y_offset=170,
            width=120,
            height=120,
        )
        detection = types.SimpleNamespace(
            targets=[types.SimpleNamespace(
                rois=[types.SimpleNamespace(type="face", rect=rect)]
            )]
        )

        node._on_detection(detection)

        self.assertIs(
            node.subscription_qos["/hobot_mono2d_body_detection"],
            module.qos_profile_sensor_data,
        )
        payload = json.loads(
            node.publishers["/servo_targets/tracking"].messages[-1].data
        )
        self.assertNotEqual(payload["targets"]["head_yaw"], 5000)
        motor = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual(motor["left"]["action"], 0)
        self.assertEqual(motor["right"]["action"], 0)

    def test_detector_waits_for_streaming_camera_status(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        initial_count = len(node.publishers["/vision_pipeline_cmd"].messages)

        node._set_tracking_mode("look_at_me")
        self.assertEqual(
            len(node.publishers["/vision_pipeline_cmd"].messages),
            initial_count,
        )

        node._on_camera_status(_FakeString('{"state":"streaming"}'))
        self.assertEqual(
            node.publishers["/vision_pipeline_cmd"].messages[-1].data,
            "start",
        )

        node._on_camera_status(_FakeString('{"state":"error"}'))
        self.assertEqual(
            node.publishers["/vision_pipeline_cmd"].messages[-1].data,
            "stop",
        )

    def test_loss_search_stops_then_disables_tracking_pipeline(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("follow_me")

        with patch.object(module.time, "monotonic", return_value=101.1):
            node._last_detection_message = 101.1
            node._control_tick()
        search_cmd = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual(search_cmd["left"]["action"], 1)
        self.assertEqual(search_cmd["right"]["action"], 2)

        with patch.object(module.time, "monotonic", return_value=105.1):
            node._last_detection_message = 105.1
            node._control_tick()
        stop_cmd = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual(stop_cmd["left"]["action"], 0)
        self.assertEqual(stop_cmd["right"]["action"], 0)
        self.assertEqual(node.mode, node.MODE_BODY_FOLLOW)

        with patch.object(module.time, "monotonic", return_value=160.1):
            node._last_detection_message = 100.0
            node._control_tick()
        self.assertEqual(node.mode, node.MODE_IDLE)
        pipeline_messages = node.publishers["/vision_pipeline_cmd"].messages
        self.assertEqual(pipeline_messages[-1].data, "stop")
        camera_messages = node.publishers["/camera_capture_cmd"].messages
        self.assertIn('"action":"acquire"', camera_messages[0].data)
        self.assertIn('"action":"release"', camera_messages[-1].data)

    def test_detector_startup_does_not_use_target_loss_timeout(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("follow_me")

        with patch.object(module.time, "monotonic", return_value=160.1):
            node._control_tick()

        self.assertEqual(node.mode, node.MODE_BODY_FOLLOW)

    def test_look_at_me_never_rotates_chassis_while_detection_is_missing(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("look_at_me")

        with patch.object(module.time, "monotonic", return_value=101.1):
            node._control_tick()

        motor = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual(motor["left"]["action"], 0)
        self.assertEqual(motor["right"]["action"], 0)

    def test_missing_detection_stage_emits_actionable_warning(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("look_at_me")

        with patch.object(module.time, "monotonic", return_value=105.1):
            node._control_tick()

        self.assertEqual(node.mode, node.MODE_FACE_FOLLOW)
        warnings = [text for level, text in node.logger.messages if level == "warning"]
        self.assertTrue(any("视觉检测话题尚无消息" in text for text in warnings))

    def test_gaze_starts_raised_and_body_dropout_does_not_bow(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        node._set_tracking_mode("look_at_me")
        self.assertEqual(
            node._tracking_controller.current_neck_pitch,
            node.GAZE_START_PITCH,
        )
        payload = json.loads(node.publishers["/servo_targets/tracking"].messages[-1].data)
        self.assertEqual(payload["targets"]["neck_bottom"],
                         node._neck_kinematics.targets(node.GAZE_START_PITCH)["neck_bottom"])
        for _ in range(100):
            node._handle_face_follow([], [(480, 480, 0.3)], 0.1)
        self.assertEqual(
            node._tracking_controller.current_neck_pitch,
            node.GAZE_START_PITCH,
        )

    def test_pitch_is_rate_limited_and_independent_of_frame_rate(self):
        module = _load_tracking_module()
        pitches = []
        for fps in (10, 30):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("look_at_me")
            for _ in range(fps):
                before = node._tracking_controller.current_neck_pitch
                node._handle_face_follow([(480, 400, 0.05)], [], 1 / fps)
                self.assertLessEqual(abs(
                    node._tracking_controller.current_neck_pitch - before
                ),
                                     node.PITCH_RATE / fps + 1e-9)
            pitches.append(node._tracking_controller.current_neck_pitch)
        self.assertAlmostEqual(*pitches, places=6)
        for _ in range(300):
            node._handle_face_follow([(480, 400, 0.05)], [], 0.1)
        self.assertEqual(
            node._tracking_controller.current_neck_pitch,
            node.GAZE_MIN_PITCH,
        )

    def test_search_uses_last_direction_and_stops_when_detector_stalls(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("follow_me")
            node._handle_body_follow([(800, 272, 0.2)], 0.1)
        with patch.object(module.time, "monotonic", return_value=101.1):
            node._last_detection_message = 101.1
            node._control_tick()
        motor = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual((motor["left"]["action"], motor["right"]["action"]), (2, 1))
        with patch.object(module.time, "monotonic", return_value=102.0):
            node._control_tick()
        motor = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual((motor["left"]["action"], motor["right"]["action"]), (0, 0))

    def test_follow_does_not_rotate_during_detector_startup(self):
        module = _load_tracking_module()
        with patch.object(module.time, "monotonic", return_value=100.0):
            node = module.WaliTrackingNode()
            node._set_tracking_mode("follow_me")
        with patch.object(module.time, "monotonic", return_value=102.0):
            node._control_tick()
        motor = json.loads(node.publishers["/motor_cmd/tracking"].messages[-1].data)
        self.assertEqual((motor["left"]["action"], motor["right"]["action"]), (0, 0))

    def test_head_targets_use_dedicated_latest_value_topic(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()

        node._publish_head_and_neck(0.5, -0.25)

        messages = node.publishers["/servo_targets/tracking"].messages
        self.assertEqual(len(messages), 1)
        payload = json.loads(messages[0].data)
        self.assertEqual(payload["targets"]["head_yaw"], 3700)
        self.assertNotIn("name", payload)

    def test_main_keeps_context_alive_for_fail_safe_shutdown(self):
        module = _load_tracking_module()
        node = Mock()
        node.MODE_IDLE = "idle"

        with (
            patch.object(module, "WaliTrackingNode", return_value=node),
            patch.object(module.rclpy, "init") as init,
            patch.object(module.rclpy, "spin", side_effect=KeyboardInterrupt),
            patch.object(module.rclpy, "shutdown") as shutdown,
            patch.object(module.signal, "signal") as install_signal,
        ):
            module.main()

        init.assert_called_once_with(args=None, signal_handler_options="no")
        install_signal.assert_called_once_with(
            module.signal.SIGTERM,
            module.signal.default_int_handler,
        )
        node._set_tracking_mode.assert_called_once_with("idle")
        node.destroy_node.assert_called_once_with()
        shutdown.assert_called_once_with()

    def test_on_detection_first_message_logs_connection_with_roi_types(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        node._set_tracking_mode("follow_me")
        detection = types.SimpleNamespace(
            targets=[types.SimpleNamespace(
                rois=[
                    types.SimpleNamespace(
                        type="body",
                        rect=types.SimpleNamespace(x_offset=10, y_offset=10, width=50, height=100),
                    ),
                    types.SimpleNamespace(
                        type="face",
                        rect=types.SimpleNamespace(x_offset=10, y_offset=10, width=0, height=20),
                    ),
                    types.SimpleNamespace(
                        type="car",
                        rect=types.SimpleNamespace(x_offset=20, y_offset=20, width=80, height=80),
                    ),
                ]
            )]
        )

        node._on_detection(detection)

        info_logs = [msg for lvl, msg in node.logger.messages if lvl == "info"]
        self.assertTrue(
            any("视觉检测链路已连通" in msg and "roi_types=['body', 'car', 'face']" in msg for msg in info_logs)
        )

    def test_on_detection_updates_last_nonempty_detection_only_for_valid_targets(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        node._set_tracking_mode("follow_me")
        node._last_nonempty_detection = 0.0

        invalid_detection = types.SimpleNamespace(
            targets=[types.SimpleNamespace(
                rois=[
                    types.SimpleNamespace(
                        type="unknown",
                        rect=types.SimpleNamespace(x_offset=0, y_offset=0, width=50, height=50),
                    ),
                    types.SimpleNamespace(
                        type="body",
                        rect=types.SimpleNamespace(x_offset=0, y_offset=0, width=0, height=50),
                    ),
                ]
            )]
        )
        with patch.object(module.time, "monotonic", return_value=50.0):
            node._on_detection(invalid_detection)
        self.assertEqual(node._last_nonempty_detection, 0.0)

        valid_detection = types.SimpleNamespace(
            targets=[types.SimpleNamespace(
                rois=[
                    types.SimpleNamespace(
                        type="body",
                        rect=types.SimpleNamespace(x_offset=10, y_offset=10, width=50, height=50),
                    ),
                ]
            )]
        )
        with patch.object(module.time, "monotonic", return_value=55.0):
            node._on_detection(valid_detection)
        self.assertEqual(node._last_nonempty_detection, 55.0)

    def test_on_detection_dispatches_body_and_face_follow(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()

        detection = types.SimpleNamespace(
            targets=[types.SimpleNamespace(
                rois=[
                    types.SimpleNamespace(
                        type="body",
                        rect=types.SimpleNamespace(x_offset=10, y_offset=10, width=50, height=100),
                    ),
                    types.SimpleNamespace(
                        type="face",
                        rect=types.SimpleNamespace(x_offset=20, y_offset=20, width=30, height=30),
                    ),
                ]
            )]
        )

        with patch.object(node, "_handle_body_follow") as mock_body:
            node._set_tracking_mode("follow_me")
            node._on_detection(detection)
            self.assertEqual(mock_body.call_count, 1)
            body_boxes = mock_body.call_args[0][0]
            self.assertEqual(len(body_boxes), 1)

        with patch.object(node, "_handle_face_follow") as mock_face:
            node._set_tracking_mode("look_at_me")
            node._on_detection(detection)
            self.assertEqual(mock_face.call_count, 1)
            face_boxes, body_boxes = mock_face.call_args[0][:2]
            self.assertEqual(len(face_boxes), 1)
            self.assertEqual(len(body_boxes), 1)

    def test_on_detection_ignores_in_idle_or_joy_override(self):
        module = _load_tracking_module()
        node = module.WaliTrackingNode()
        detection = types.SimpleNamespace(targets=[])

        # Idle mode
        self.assertEqual(node.mode, node.MODE_IDLE)
        node._on_detection(detection)
        self.assertEqual(node._last_detection_message, 0.0)

        # Joy override
        node._set_tracking_mode("follow_me")
        node._joy_override = True
        node._on_detection(detection)
        self.assertEqual(node._last_detection_message, 0.0)


if __name__ == "__main__":
    unittest.main()
