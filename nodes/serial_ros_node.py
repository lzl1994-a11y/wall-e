#!/usr/bin/env python3
"""串口通信节点：唯一的串口持有者。

订阅 Topic 并透传下位机：
  /screen_dialog → 屏幕文字（you: / ai:）
  /tft_cmd       → TFT 控制指令（eyeaction:...）
  /pca9685_raw   → PCA9685 15 通道原始值（由 hardware_bridge_node 产出）

动作分发由 sequence_ros_node 负责。serial_mcu 模式下运动数据由
hardware_bridge_node 发送；ubuntu_i2c 模式下本节点只负责屏幕通信。
  本节点只做串口透传；眼睛配置和 NETCFG 仅做必要的协议路由与回包关联。
"""

import json
import threading
import time
import uuid
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from services.hardware.serial_bridge import SerialBridge
from services.hardware.esp32_netcfg import (
    Esp32NetworkConfigurator,
    NetworkConfigError,
    load_saved_network_settings,
    resolve_session_network_settings,
    validate_network_payload,
)
from services.hardware.esp32_netcfg_rpc import REQUEST_TOPIC, RESPONSE_TOPIC
from services.audio.music_protocol import MUSIC_STATE_TOPIC, decode_music_state
from services.speech.tts_protocol import decode_turn_end
from services.hardware.eyeconfig_rpc import (
    EYE_REQUEST_TOPIC,
    EYE_RESPONSE_TOPIC,
    EYE_STATUS_TOPIC,
    EyeConfigError,
    parse_eye_response,
    validate_eye_command,
)


class SerialNode(Node):
    def __init__(self):
        super().__init__('walle_serial_node')

        self.get_logger().info('Serial bridge node starting...')
        self.bridge = SerialBridge(device_name="WALL_E_TFT")
        self.netcfg = Esp32NetworkConfigurator()
        self._netcfg_request_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._tft_preview_ready = threading.Event()
        self._music_active = False
        self._last_user_turn_id = ""
        self._eye_request_lock = threading.Lock()
        self._eye_response_condition = threading.Condition()
        self._eye_events = deque(maxlen=64)
        self._eye_event_sequence = 0
        self._last_eye_state = {}
        self._eye_blink_ms = None
        self._eye_effective_blink_ms = None
        self._eye_speaking = False
        self._eye_blink_revision = 0
        self._eye_blink_paused = False

        if not self.bridge.ser:
            self.get_logger().error('Serial bridge connection failed; check hardware connection.')

        # 订阅 Topic
        self.create_subscription(String, 'screen_dialog', self.screen_dialog_callback, 10)
        self.create_subscription(String, 'tft_cmd', self.tft_cmd_callback, 10)
        self.create_subscription(String, MUSIC_STATE_TOPIC, self._on_music_state, 10)
        # Motion state is latest-wins. Keeping ten stale states here causes a
        # visible catch-up burst after an exclusive serial transaction.
        self.create_subscription(
            String,
            'pca9685_raw',
            self.pca9685_callback,
            QoSProfile(depth=1),
        )
        self._netcfg_response_publisher = self.create_publisher(String, RESPONSE_TOPIC, 10)
        self._netcfg_status_publisher = self.create_publisher(
            String,
            "/esp32_netcfg_status",
            QoSProfile(
                # Retain both startup phases long enough for an audio process
                # that initializes slightly later than this node.
                depth=4,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.create_subscription(String, REQUEST_TOPIC, self.netcfg_request_callback, 10)
        self._eye_response_publisher = self.create_publisher(String, EYE_RESPONSE_TOPIC, 10)
        self._eye_status_publisher = self.create_publisher(String, EYE_STATUS_TOPIC, 20)
        self.create_subscription(String, EYE_REQUEST_TOPIC, self.eyeconfig_request_callback, 10)
        self.create_subscription(String, 'tts_text', self._on_eye_tts, 10)
        self.create_subscription(String, 'llm_busy', self._on_eye_playback, 10)
        self.bridge.add_line_listener(self._on_serial_line)
        self._tft_ready_subscription = self.create_subscription(
            String,
            "tft_preview_ready",
            self._on_tft_preview_ready,
            10,
        )

        self.get_logger().info('Serial ROS node is online (sole serial owner).')
        self._startup_netcfg_thread = threading.Thread(
            target=self._apply_saved_network_on_start,
            name="esp32-netcfg-startup",
            daemon=True,
        )
        self._startup_netcfg_thread.start()

    def _on_tft_preview_ready(self, _message):
        self._tft_preview_ready.set()

    def _on_music_state(self, message):
        state = decode_music_state(message.data)
        if state is not None:
            self._music_active = state["state"] in {"loading", "playing"}

    def _apply_saved_network_on_start(self):
        """Push a fresh RAM-only Wi-Fi/TCP session after every process start."""
        settings = None
        status_request_id = uuid.uuid4().hex
        try:
            saved = load_saved_network_settings()
        except NetworkConfigError as exc:
            self.get_logger().error(f"启动时 ESP32 网络配置无效: {exc}")
            self._publish_netcfg_status("failed", detail=str(exc), request_id=status_request_id)
            return
        if saved is None:
            detail = "未配置 esp32_network，无法建立 ESP32 网络会话"
            self.get_logger().info(detail)
            self._publish_netcfg_status("failed", detail=detail, request_id=status_request_id)
            return
        try:
            settings, sources = resolve_session_network_settings(saved)
        except NetworkConfigError as exc:
            self.get_logger().error(f"启动时解析当前网络失败: {exc}")
            self._publish_netcfg_status("failed", detail=str(exc), request_id=status_request_id)
            return
        deadline = time.monotonic() + 30.0
        while not self._tft_preview_ready.is_set():
            if self._shutdown_event.wait(0.2):
                return
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    "等待 TFT TCP 服务监听就绪超时，未向 ESP32 应用网络配置"
                )
                self._publish_netcfg_status(
                    "failed",
                    settings=settings,
                    detail="等待 TFT TCP 服务监听就绪超时",
                    request_id=status_request_id,
                )
                return
        if not self._netcfg_request_lock.acquire(timeout=5.0):
            self.get_logger().warning("ESP32 网络配置正忙，跳过启动时重复应用")
            self._publish_netcfg_status(
                "failed",
                settings=settings,
                detail="ESP32 网络配置正忙",
                request_id=status_request_id,
            )
            return
        try:
            # QUERY is deliberately retained as a protocol/version health check,
            # but v2 session data is always resent because firmware keeps it in
            # RAM only and may have rebooted independently of the upper host.
            self.bridge.run_exclusive(
                lambda stream: self.netcfg.query(stream=stream)
            )
            self.get_logger().info(
                f"启动时下发 ESP32 会话网络: {settings.host}:{settings.port} "
                f"(ssid={sources['ssid_source']}, host={sources['host_source']})"
            )
            result = self.bridge.run_exclusive(
                lambda stream: self.netcfg.save_and_apply(
                    settings,
                    stream=stream,
                    on_phase=lambda state: self._publish_netcfg_status(
                        state, settings=settings, request_id=status_request_id
                    ),
                )
            )
            self.get_logger().info(
                f"启动时 ESP32 网络配置成功: SET #{result['set_seq']}, "
                f"APPLY #{result['apply_seq']}"
            )
        except (NetworkConfigError, RuntimeError) as exc:
            self.get_logger().error(f"启动时 ESP32 网络配置失败: {exc}")
            self._publish_netcfg_status(
                "failed", settings=settings, detail=str(exc), request_id=status_request_id
            )
        except Exception:
            self.get_logger().error("启动时 ESP32 网络配置串口通信失败")
            self._publish_netcfg_status(
                "failed",
                settings=settings,
                detail="ESP32 网络配置串口通信失败",
                request_id=status_request_id,
            )
        finally:
            self._netcfg_request_lock.release()

    def _publish_netcfg_status(
        self, state, *, settings=None, detail="", request_id=""
    ):
        """Publish a secret-free latest status for UI and prompt-audio consumers."""
        body = {
            "state": state,
            "host": settings.host if settings is not None else "",
            "port": settings.port if settings is not None else 0,
            "detail": detail,
            "request_id": request_id,
            "published_at": time.time(),
        }
        self._netcfg_status_publisher.publish(
            String(data=json.dumps(body, ensure_ascii=False, separators=(",", ":")))
        )

    # ------------------------------------------------------------------
    # screen_dialog: 屏幕文字
    # ------------------------------------------------------------------
    def screen_dialog_callback(self, msg):
        """Send a complete turn to the lower screen in one callback."""
        try:
            dialog = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f"screen_dialog JSON parse failed: {msg.data}")
            return

        turn_id = dialog.get("turn_id", "")
        corrected_text = (dialog.get("corrected_text") or "").strip()
        ai_text = (dialog.get("ai_text") or "").strip()

        music_hint = self._music_surface_hint(dialog.get("actions"))
        preserve_music = self._music_active if music_hint is None else music_hint
        if preserve_music:
            if ai_text:
                self.bridge.send_raw("eyeaction:talk\n", wake_screen=False)
            return

        user_already_sent = bool(turn_id) and turn_id == getattr(
            self, "_last_user_turn_id", ""
        )
        if corrected_text and not user_already_sent:
            payload = f"you:{corrected_text}\n"
            if self.bridge.send_raw(payload):
                if turn_id:
                    self._last_user_turn_id = turn_id
                self.get_logger().info(f'[{turn_id}] Sent user text -> {payload.strip()}')

        if ai_text:
            self.bridge.send_raw("openchat:1\n")
            self.bridge.send_raw("eyeaction:talk\n")
            payload = f"ai:{ai_text}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'[{turn_id}] Sent AI text -> {payload.strip()}')

    @staticmethod
    def _music_surface_hint(actions):
        """Return the last confirmed music navigation intent in a dialog."""
        hint = None
        for action in actions if isinstance(actions, list) else []:
            if not isinstance(action, dict) or action.get("name") != "control_music":
                continue
            if action.get("status") != "completed":
                continue
            arguments = action.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            if arguments.get("action") == "play":
                hint = True
            elif arguments.get("action") == "stop":
                hint = False
        return hint

    def you_callback(self, msg):
        if self._music_active:
            return
        payload = f"you:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent user text -> {payload.strip()}')

    def ai_callback(self, msg):
        """Handle a full AI response from the legacy topic."""
        if self._music_active:
            self.bridge.send_raw("eyeaction:talk\n", wake_screen=False)
            return
        self.bridge.send_raw("openchat:1\n")
        self.bridge.send_raw("eyeaction:talk\n")
        payload = f"ai:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent AI text -> {payload.strip()}')

    # ------------------------------------------------------------------
    # tft_cmd: 表情控制指令
    # ------------------------------------------------------------------
    def tft_cmd_callback(self, msg):
        if self.bridge.send_raw(msg.data, wake_screen=not self._music_active):
            self.get_logger().debug(f'[Serial] TFT cmd forwarded: {msg.data.strip()}')

    # ------------------------------------------------------------------
    # eyeconfig_request: web eye configuration -> shared serial owner
    # ------------------------------------------------------------------
    def _on_eye_tts(self, message):
        text = (message.data or "").strip()
        if text and decode_turn_end(text) is None:
            self._set_eye_speaking(True)

    def _on_eye_playback(self, message):
        if message.data == "idle":
            self._set_eye_speaking(False)

    def _set_eye_speaking(self, speaking):
        with self._eye_response_condition:
            if self._shutdown_event.is_set() or self._eye_speaking == speaking:
                return
            self._eye_speaking = speaking
            self._eye_blink_revision += 1
            revision = self._eye_blink_revision
        self._eye_status_publisher.publish(String(data=json.dumps({"speaking": speaking})))
        threading.Thread(target=self._sync_eye_blink, args=(revision,),
                         name="eye-speaking-blink", daemon=True).start()

    def _sync_eye_blink(self, revision):
        # Share ACK correlation with web commands; never write competing eye
        # commands from the ROS callback or the serial reader thread.
        if not self._eye_request_lock.acquire(timeout=3.0):
            self.get_logger().error("等待眼睛配置事务超时，眨眼状态未同步")
            return
        try:
            with self._eye_response_condition:
                if revision != self._eye_blink_revision or self._shutdown_event.is_set():
                    return
            if self._eye_blink_ms is None:
                result = self._exchange_eye_command("eyeconfig:query", "state")
                if not result["ok"]:
                    raise RuntimeError(result["error"])
                if self._eye_blink_ms is None:
                    raise RuntimeError("设备未回显 blinkMs，不能安全暂停眨眼")
            with self._eye_response_condition:
                period = 0 if self._eye_speaking else self._eye_blink_ms
                if period == self._eye_effective_blink_ms:
                    return
            result = self._exchange_eye_command(f"eyeconfig:blinkMs={period}", "ack")
            if not result["ok"]:
                raise RuntimeError(result["error"])
            with self._eye_response_condition:
                self._eye_effective_blink_ms = period
                self._eye_blink_paused = period == 0 and self._eye_blink_ms > 0
        except RuntimeError as exc:
            self.get_logger().error(f"眨眼状态同步失败: {exc}")
        finally:
            self._eye_request_lock.release()

    def _on_serial_line(self, line):
        """Observe EYE replies without opening a second serial connection."""
        event = parse_eye_response(line)
        if event is None:
            return
        with self._eye_response_condition:
            self._eye_event_sequence += 1
            sequence = self._eye_event_sequence
            event["sequence"] = sequence
            self._eye_events.append((sequence, event))
            if event.get("kind") == "state":
                fields = event.get("fields") or {}
                if "blinkMs" in fields:
                    self._eye_effective_blink_ms = fields["blinkMs"]
                    if self._eye_blink_ms is None or (not self._eye_speaking and not self._eye_blink_paused):
                        self._eye_blink_ms = fields["blinkMs"]
                    # Surface the user configuration, not a temporary TTS pause.
                    fields["blinkMs"] = self._eye_blink_ms
                self._last_eye_state.update(event.get("fields") or {})
            self._eye_response_condition.notify_all()
        self._eye_status_publisher.publish(
            String(data=json.dumps({"event": event, "speaking": self._eye_speaking}, ensure_ascii=False, separators=(",", ":")))
        )

    def eyeconfig_request_callback(self, message):
        try:
            request = json.loads(message.data)
            if not isinstance(request, dict):
                return
            request_id = request.get("request_id")
            command = validate_eye_command(request.get("command"))
            expected = request.get("expected")
        except (AttributeError, TypeError, json.JSONDecodeError, EyeConfigError) as exc:
            request_id = request.get("request_id") if isinstance(locals().get("request"), dict) else ""
            self._publish_eye_response(
                request_id,
                ok=False,
                error=str(exc) or "眼睛命令格式无效",
            )
            return
        if not isinstance(request_id, str) or not request_id or expected not in {"state", "ack"}:
            return
        threading.Thread(
            target=self._run_eyeconfig_request,
            args=(request_id, command, expected),
            name="eyeconfig-request",
            daemon=True,
        ).start()

    def _wait_for_eye_event(self, after_sequence, expected, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self._eye_response_condition:
            while True:
                for sequence, event in self._eye_events:
                    if sequence <= after_sequence:
                        continue
                    kind = event.get("kind")
                    if kind == "err" or (expected == "state" and kind == "state") or (
                        expected == "ack" and kind == "ok"
                    ):
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._shutdown_event.is_set():
                    return None
                self._eye_response_condition.wait(remaining)

    def _exchange_eye_command(self, command, expected):
        """One transaction; caller holds the existing eye request lock."""
        with self._eye_response_condition:
            after_sequence = self._eye_event_sequence
        if not self.bridge.send_raw(command + "\n", wake_screen=False):
            raise RuntimeError("串口未连接，眼睛命令未发送")
        event = self._wait_for_eye_event(after_sequence, expected)
        if event is None:
            raise RuntimeError("等待眼睛设备响应超时")
        response = {"ok": event.get("kind") != "err", "event": event}
        if not response["ok"]:
            response["error"] = event.get("message") or "眼睛设备拒绝了命令"
        elif expected == "state":
            with self._eye_response_condition:
                response["state"] = dict(self._last_eye_state)
        elif event.get("fields"):
            response["state"] = dict(event["fields"])
        return response

    def _run_eyeconfig_request(self, request_id, command, expected):
        if not self._eye_request_lock.acquire(timeout=3.0):
            self._publish_eye_response(request_id, ok=False, error="等待眼睛配置事务超时")
            return
        response = {"ok": False}
        try:
            with self._eye_response_condition:
                if command.startswith("eyeconfig:scale="):
                    scale = float(command.split("=", 1)[1])
                    if not self._last_eye_state.get("minScale", 0.4) <= scale <= self._last_eye_state.get("maxScale", 1.5):
                        raise RuntimeError("缩放超出设备当前 minScale/maxScale 范围")
                requested_period = int(command.split("=", 1)[1]) if command.startswith("eyeconfig:blinkMs=") else None
                wire_command = "eyeconfig:blinkMs=0" if requested_period is not None and self._eye_speaking else command
            response = self._exchange_eye_command(wire_command, expected)
            if response["ok"] and requested_period is not None:
                with self._eye_response_condition:
                    self._eye_blink_ms = requested_period
                    self._eye_effective_blink_ms = int(wire_command.split("=", 1)[1])
                    self._eye_blink_paused = self._eye_effective_blink_ms == 0 and requested_period > 0
                    self._last_eye_state["blinkMs"] = requested_period
                response["state"] = {"blinkMs": requested_period}
        except Exception as exc:
            response["error"] = str(exc)
        finally:
            try:
                response["speaking"] = self._eye_speaking
                self._publish_eye_response(request_id, **response)
            finally:
                self._eye_request_lock.release()

    def _publish_eye_response(self, request_id, *, ok=False, error="", event=None, state=None, speaking=None):
        body = {"request_id": request_id, "ok": bool(ok)}
        if speaking is not None:
            body["speaking"] = speaking
        if error:
            body["error"] = str(error)
        if isinstance(event, dict):
            body["event"] = event
        if isinstance(state, dict):
            body["state"] = state
        self._eye_response_publisher.publish(
            String(data=json.dumps(body, ensure_ascii=False, separators=(",", ":")))
        )

    # ------------------------------------------------------------------
    # pca9685_raw: 硬件 15 通道原始值（原 hardware_bridge_node 直接写串口）
    # ------------------------------------------------------------------
    def pca9685_callback(self, msg):
        payload = msg.data + '\n'
        # NETCFG may temporarily own the serial stream. Never block the ROS
        # executor or queue stale servo positions behind that transaction.
        if self.bridge.send_raw(payload, block=False, wake_screen=not self._music_active):
            self.get_logger().debug(f'[Serial] PCA9685 forwarded ({len(msg.data)} bytes)')

    # ------------------------------------------------------------------
    # ESP32 network configuration: Web -> ROS RPC -> this sole serial owner.
    # ------------------------------------------------------------------
    def netcfg_request_callback(self, msg):
        """Dispatch the potentially 65+ second APPLY without blocking ROS spin."""
        try:
            request = json.loads(msg.data)
            if not isinstance(request, dict):
                return
            request_id = request.get("request_id")
            operation = request.get("operation")
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(request_id, str) or operation not in {"save_and_apply", "query"}:
            return
        if not self._netcfg_request_lock.acquire(blocking=False):
            response = String()
            response.data = json.dumps(
                {"request_id": request_id, "ok": False, "error": "已有 ESP32 网络配置操作正在执行"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._netcfg_response_publisher.publish(response)
            return
        # Never log request data: it can include Wi-Fi passwords.
        threading.Thread(
            target=self._run_netcfg_request,
            args=(request_id, operation, request.get("payload")),
            name="esp32-netcfg",
            daemon=True,
        ).start()

    def _run_netcfg_request(self, request_id, operation, payload):
        response = {"request_id": request_id, "ok": False}
        try:
            if operation == "save_and_apply":
                if not isinstance(payload, dict):
                    raise NetworkConfigError("网络配置请求格式错误")
                # Validate before taking the physical serial connection; a bad
                # browser request must not reset a healthy screen connection.
                settings = validate_network_payload(payload)
                data = self.bridge.run_exclusive(
                    lambda stream: self.netcfg.save_and_apply(
                        settings,
                        stream=stream,
                        on_phase=lambda state: self._publish_netcfg_status(
                            state, settings=settings, request_id=request_id
                        ),
                    )
                )
            else:
                data = self.bridge.run_exclusive(lambda stream: self.netcfg.query(stream=stream))
            response.update(ok=True, data=data)
        except (NetworkConfigError, RuntimeError) as exc:
            response["error"] = str(exc)
            if operation == "save_and_apply":
                self._publish_netcfg_status(
                    "failed",
                    settings=locals().get("settings"),
                    detail=str(exc),
                    request_id=request_id,
                )
        except Exception:
            # Do not expose or log request contents; serial details are not useful
            # to the browser and could accidentally include sensitive input.
            response["error"] = "ESP32 网络配置串口通信失败"
            if operation == "save_and_apply":
                self._publish_netcfg_status(
                    "failed",
                    settings=locals().get("settings"),
                    detail=response["error"],
                    request_id=request_id,
                )
        try:
            message = String()
            message.data = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            self._netcfg_response_publisher.publish(message)
        finally:
            self._netcfg_request_lock.release()

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info('Closing serial bridge...')
        self._shutdown_event.set()
        if hasattr(self, 'bridge'):
            self.bridge.remove_line_listener(self._on_serial_line)
        if hasattr(self, 'bridge'):
            self.bridge.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
