import unittest
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from services.esp32_netcfg import (
    APPLY_FINAL_TIMEOUT_SECONDS,
    Esp32NetworkConfigurator,
    NetworkConfigError,
    decode_urlsafe_base64,
    encode_urlsafe_base64,
    load_saved_network_settings,
    network_settings_to_payload,
    resolve_session_network_settings,
    validate_network_payload,
)


class FakeSerial:
    def __init__(self, incoming=b""):
        self.incoming = bytearray(incoming)
        self.writes = []
        self.closed = False

    def write(self, value):
        self.writes.append(bytes(value))
        return len(value)

    def read(self, _count=1):
        if not self.incoming:
            return b""
        value = bytes(self.incoming[:1])
        del self.incoming[:1]
        return value

    def flush(self):
        pass

    def close(self):
        self.closed = True


class Esp32NetworkProtocolTests(unittest.TestCase):
    def test_urlsafe_base64_is_utf8_and_unpadded(self):
        self.assertEqual(encode_urlsafe_base64("MyWiFi"), "TXlXaUZp")
        self.assertEqual(encode_urlsafe_base64("中文"), "5Lit5paH")
        self.assertNotIn("=", encode_urlsafe_base64("a"))
        self.assertEqual(decode_urlsafe_base64("5Lit5paH", "SSID"), "中文")
        with self.assertRaises(NetworkConfigError):
            decode_urlsafe_base64("YQ==", "SSID")

    def test_validation_uses_utf8_bytes_and_requires_exactly_three_pairs(self):
        payload = {"wifi": [{"ssid": "中" * 11, "password": ""}] * 3, "host": "host", "port": 9000}
        with self.assertRaisesRegex(NetworkConfigError, "32"):
            validate_network_payload(payload)
        with self.assertRaisesRegex(NetworkConfigError, "固定的 3"):
            validate_network_payload({"wifi": [], "host": "host", "port": 9000})
        with self.assertRaisesRegex(NetworkConfigError, "同时留空"):
            validate_network_payload({"wifi": [{"ssid": "", "password": "secret"}] + [{"ssid": "", "password": ""}] * 2, "host": "host", "port": 9000})

    def test_validated_settings_repr_redacts_password(self):
        settings = validate_network_payload(
            {"wifi": [{"ssid": "MyWiFi", "password": "do-not-print"}] + [{"ssid": "", "password": ""}] * 2, "host": "host", "port": 9000}
        )
        self.assertIn("MyWiFi", repr(settings))
        self.assertNotIn("do-not-print", repr(settings))

    def test_saved_full_credentials_can_be_loaded_for_startup_without_repr_leak(self):
        payload = {
            "wifi": [
                {"ssid": "MyWiFi", "password": "do-not-print"},
                {"ssid": "", "password": ""},
                {"ssid": "", "password": ""},
            ],
            "host": "192.168.0.155",
            "port": 9000,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                yaml.safe_dump({"esp32_network": payload}, allow_unicode=True),
                encoding="utf-8",
            )
            settings = load_saved_network_settings(path)

        self.assertEqual(network_settings_to_payload(settings), payload)
        self.assertNotIn("do-not-print", repr(settings))

    def test_set_then_apply_ignores_logs_and_other_sequences(self):
        incoming = (
            b"booting...\r\n"
            b"NETCFG:RESULT:999|SET|0|0\n"
            b"NETCFG:RESULT:1001|SET|0|0\r\n"
            b"normal log\n"
            b"NETCFG:RESULT:1002|APPLY|1|0\n"
            b"NETCFG:RESULT:1002|APPLY|2|0\r\n"
        )
        stream = FakeSerial(incoming)
        client = Esp32NetworkConfigurator(serial_factory=lambda *_args, **_kwargs: stream, port_resolver=lambda: "test")
        client._next_seq = iter((1001, 1002)).__next__
        result = client.save_and_apply(
            {"wifi": [{"ssid": "MyWiFi", "password": "12345678"}] + [{"ssid": "", "password": ""}] * 2, "host": "192.168.1.100", "port": 9000}
        )
        self.assertEqual(result, {"set_seq": 1001, "apply_seq": 1002, "result": "applied"})
        self.assertEqual(stream.writes[0], b"netcfg:set:1001|2|TXlXaUZp|MTIzNDU2Nzg|||||MTkyLjE2OC4xLjEwMA|9000\r\n")
        self.assertEqual(stream.writes[1], b"netcfg:apply:1002|2\r\n")
        self.assertTrue(stream.closed)

    def test_apply_acceptance_is_not_reported_as_success(self):
        stream = FakeSerial(b"NETCFG:RESULT:1|SET|0|0\nNETCFG:RESULT:2|APPLY|1|0\nNETCFG:RESULT:2|APPLY|6|0\n")
        client = Esp32NetworkConfigurator(serial_factory=lambda *_args, **_kwargs: stream, port_resolver=lambda: "test")
        client._next_seq = iter((1, 2)).__next__
        with self.assertRaisesRegex(NetworkConfigError, "HELLO"):
            client.save_and_apply({"wifi": [{"ssid": "a", "password": ""}] + [{"ssid": "", "password": ""}] * 2, "host": "host", "port": 9000})

    def test_apply_sequence_is_distinct_when_random_source_collides(self):
        stream = FakeSerial(b"NETCFG:RESULT:9|SET|0|0\nNETCFG:RESULT:10|APPLY|1|0\nNETCFG:RESULT:10|APPLY|2|0\n")
        client = Esp32NetworkConfigurator(serial_factory=lambda *_args, **_kwargs: stream, port_resolver=lambda: "test")
        client._next_seq = iter((9, 9, 10)).__next__
        client.save_and_apply({"wifi": [{"ssid": "a", "password": ""}] + [{"ssid": "", "password": ""}] * 2, "host": "host", "port": 9000})
        self.assertEqual(stream.writes[1], b"netcfg:apply:10|2\r\n")

    def test_query_decodes_response_and_never_includes_passwords(self):
        stream = FakeSerial(b"plain log\rNETCFG:STATUS:8|2|2|1|TXlXaUZp|||MTkyLjE2OC4xLjEwMA|9000\n")
        client = Esp32NetworkConfigurator(serial_factory=lambda *_args, **_kwargs: stream, port_resolver=lambda: "test")
        client._next_seq = lambda: 8
        status = client.query()
        self.assertEqual(status["wifi"], [{"ssid": "MyWiFi"}, {"ssid": ""}, {"ssid": ""}])
        self.assertEqual(status["host"], "192.168.1.100")
        self.assertFalse(status["active_from_nvs"])
        self.assertTrue(status["candidate_present"])
        self.assertNotIn("password", repr(status))
        self.assertEqual(stream.writes, [b"netcfg:query:8|2\r\n"])

    def test_query_matching_result_error_is_reported_without_waiting_for_status(self):
        stream = FakeSerial(b"NETCFG:RESULT:8|QUERY|3|2\n")
        client = Esp32NetworkConfigurator(serial_factory=lambda *_args, **_kwargs: stream, port_resolver=lambda: "test")
        client._next_seq = lambda: 8
        with self.assertRaisesRegex(NetworkConfigError, "QUERY.*字段"):
            client.query()

    def test_final_apply_timeout_is_at_least_65_seconds(self):
        self.assertGreaterEqual(APPLY_FINAL_TIMEOUT_SECONDS, 65.0)

    def test_phase_callback_fires_only_after_set_ack_and_final_success(self):
        stream = FakeSerial(
            b"NETCFG:RESULT:1|SET|0|0\n"
            b"NETCFG:RESULT:2|APPLY|1|0\n"
            b"NETCFG:RESULT:2|APPLY|2|0\n"
        )
        client = Esp32NetworkConfigurator(
            serial_factory=lambda *_args, **_kwargs: stream,
            port_resolver=lambda: "test",
        )
        client._next_seq = iter((1, 2)).__next__
        phases = []
        client.save_and_apply(
            {"wifi": [{"ssid": "a", "password": ""}] + [{"ssid": "", "password": ""}] * 2, "host": "192.168.1.2", "port": 9000},
            on_phase=phases.append,
        )
        self.assertEqual(phases, ["configuring", "connected"])

    def test_session_resolution_promotes_live_ssid_and_live_host(self):
        saved = validate_network_payload(
            {
                "wifi": [
                    {"ssid": "backup", "password": "backup-secret"},
                    {"ssid": "live", "password": "live-secret"},
                    {"ssid": "", "password": ""},
                ],
                "host": "192.168.1.9",
                "port": 9000,
            }
        )
        resolved, sources = resolve_session_network_settings(
            saved,
            ssid_detector=lambda: "live",
            host_detector=lambda: "10.0.0.8",
        )
        self.assertEqual(resolved.wifi[0].ssid, "live")
        self.assertEqual(resolved.wifi[0].password, "live-secret")
        self.assertEqual(resolved.host, "10.0.0.8")
        self.assertEqual(sources, {"ssid_source": "detected", "host_source": "detected"})

    def test_session_resolution_uses_explicit_fallbacks_when_detection_unavailable(self):
        saved = validate_network_payload(
            {"wifi": [{"ssid": "fallback", "password": "secret"}] + [{"ssid": "", "password": ""}] * 2, "host": "192.168.1.9", "port": 9000}
        )
        resolved, sources = resolve_session_network_settings(
            saved,
            ssid_detector=lambda: None,
            host_detector=lambda: None,
        )
        self.assertEqual(resolved.host, "192.168.1.9")
        self.assertEqual(sources["host_source"], "configured_fallback")

    def test_session_resolution_rejects_unknown_live_ssid_and_unsafe_host(self):
        saved = validate_network_payload(
            {"wifi": [{"ssid": "known", "password": "secret"}] + [{"ssid": "", "password": ""}] * 2, "host": "127.0.0.1", "port": 9000}
        )
        with self.assertRaisesRegex(NetworkConfigError, "未在"):
            resolve_session_network_settings(
                saved,
                ssid_detector=lambda: "unknown",
                host_detector=lambda: "192.168.1.2",
            )
        with self.assertRaisesRegex(NetworkConfigError, "非回环"):
            resolve_session_network_settings(
                saved,
                ssid_detector=lambda: "known",
                host_detector=lambda: None,
            )


class Esp32NetworkRpcSafetyTests(unittest.TestCase):
    class _GraphNode:
        def __init__(self, subscribers=0, publishers=0):
            self.subscribers = subscribers
            self.publishers = publishers

        def count_subscribers(self, topic):
            self.last_subscriber_topic = topic
            return self.subscribers

        def count_publishers(self, topic):
            self.last_publisher_topic = topic
            return self.publishers

    class _Message:
        def __init__(self, data):
            self.data = data

    def _bare_rpc(self):
        from services.esp32_netcfg_rpc import Esp32NetworkRpcClient

        rpc = object.__new__(Esp32NetworkRpcClient)
        rpc._closed = threading.Event()
        rpc._monotonic = time.monotonic
        rpc._discovery_timeout_seconds = 0.02
        rpc._lock = threading.Lock()
        rpc._pending = {}
        return rpc

    def test_ros_discovery_requires_both_request_subscriber_and_response_publisher(self):
        from services.esp32_netcfg_rpc import REQUEST_TOPIC, RESPONSE_TOPIC

        rpc = self._bare_rpc()
        rpc._node = self._GraphNode(subscribers=1, publishers=1)
        self.assertTrue(rpc._wait_for_serial_owner(time.monotonic() + 0.1))
        self.assertEqual(rpc._node.last_subscriber_topic, REQUEST_TOPIC)
        self.assertEqual(rpc._node.last_publisher_topic, RESPONSE_TOPIC)
        rpc._node.publishers = 0
        started = time.monotonic()
        self.assertFalse(rpc._wait_for_serial_owner(time.monotonic() + 0.1))
        self.assertLess(time.monotonic() - started, 0.1)

    def test_malformed_ros_response_does_not_break_pending_request(self):
        rpc = self._bare_rpc()
        event = threading.Event()
        result = {}
        rpc._pending["request-1"] = (event, result)
        rpc._on_response(self._Message("[]"))
        rpc._on_response(self._Message("null"))
        rpc._on_response(self._Message("not-json"))
        self.assertFalse(event.is_set())
        rpc._on_response(self._Message('{"request_id":"request-1","ok":true,"data":{}}'))
        self.assertTrue(event.is_set())
        self.assertTrue(result["ok"])
