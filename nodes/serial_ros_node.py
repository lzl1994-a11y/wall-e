#!/usr/bin/env python3
"""串口通信节点：唯一的串口持有者。

订阅 Topic 并透传下位机：
  /screen_dialog → 屏幕文字（you: / ai:）
  /tft_cmd       → TFT 控制指令（eyeaction:...）
  /pca9685_raw   → PCA9685 15 通道原始值（由 hardware_bridge_node 产出）

动作分发由 sequence_ros_node 负责。serial_mcu 模式下运动数据由
hardware_bridge_node 发送；ubuntu_i2c 模式下本节点只负责屏幕通信。
本节点只做串口透传，不做任何业务解析。
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

from services.serial_bridge import SerialBridge
from services.esp32_netcfg import (
    Esp32NetworkConfigurator,
    NetworkConfigError,
    load_saved_network_settings,
    network_settings_match_status,
    validate_network_payload,
)
from services.esp32_netcfg_rpc import REQUEST_TOPIC, RESPONSE_TOPIC


class SerialNode(Node):
    def __init__(self):
        super().__init__('walle_serial_node')

        self.get_logger().info('Serial bridge node starting...')
        self.bridge = SerialBridge(device_name="WALL_E_TFT")
        self.netcfg = Esp32NetworkConfigurator()
        self._netcfg_request_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._tft_preview_ready = threading.Event()

        if not self.bridge.ser:
            self.get_logger().error('Serial bridge connection failed; check hardware connection.')

        # 订阅 Topic
        self.create_subscription(String, 'screen_dialog', self.screen_dialog_callback, 10)
        self.create_subscription(String, 'tft_cmd', self.tft_cmd_callback, 10)
        # Motion state is latest-wins. Keeping ten stale states here causes a
        # visible catch-up burst after an exclusive serial transaction.
        self.create_subscription(
            String,
            'pca9685_raw',
            self.pca9685_callback,
            QoSProfile(depth=1),
        )
        self._netcfg_response_publisher = self.create_publisher(String, RESPONSE_TOPIC, 10)
        self.create_subscription(String, REQUEST_TOPIC, self.netcfg_request_callback, 10)
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

    def _apply_saved_network_on_start(self):
        """Synchronize the retained full Wi-Fi/TCP configuration after startup."""
        try:
            settings = load_saved_network_settings()
        except NetworkConfigError as exc:
            self.get_logger().error(f"启动时 ESP32 网络配置无效: {exc}")
            return
        if settings is None:
            self.get_logger().info("未保存 ESP32 网络配置，跳过启动时 SET/APPLY")
            return
        deadline = time.monotonic() + 30.0
        while not self._tft_preview_ready.is_set():
            if self._shutdown_event.wait(0.2):
                return
            if time.monotonic() >= deadline:
                self.get_logger().error(
                    "等待 TFT TCP 服务监听就绪超时，未向 ESP32 应用网络配置"
                )
                return
        if not self._netcfg_request_lock.acquire(timeout=5.0):
            self.get_logger().warning("ESP32 网络配置正忙，跳过启动时重复应用")
            return
        try:
            status = self.bridge.run_exclusive(
                lambda stream: self.netcfg.query(stream=stream)
            )
            if network_settings_match_status(settings, status):
                self.get_logger().info(
                    "ESP32 网络配置已一致，跳过启动时 SET/APPLY"
                )
                return
            self.get_logger().info(
                f"启动时同步 ESP32 图像服务器: {settings.host}:{settings.port}"
            )
            result = self.bridge.run_exclusive(
                lambda stream: self.netcfg.save_and_apply(settings, stream=stream)
            )
            self.get_logger().info(
                f"启动时 ESP32 网络配置成功: SET #{result['set_seq']}, "
                f"APPLY #{result['apply_seq']}"
            )
        except (NetworkConfigError, RuntimeError) as exc:
            self.get_logger().error(f"启动时 ESP32 网络配置失败: {exc}")
        except Exception:
            self.get_logger().error("启动时 ESP32 网络配置串口通信失败")
        finally:
            self._netcfg_request_lock.release()

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

        if corrected_text:
            payload = f"you:{corrected_text}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'[{turn_id}] Sent user text -> {payload.strip()}')

        if ai_text:
            self.bridge.send_raw("openchat:1\n")
            self.bridge.send_raw("eyeaction:talk\n")
            payload = f"ai:{ai_text}\n"
            if self.bridge.send_raw(payload):
                self.get_logger().info(f'[{turn_id}] Sent AI text -> {payload.strip()}')

    def you_callback(self, msg):
        payload = f"you:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent user text -> {payload.strip()}')

    def ai_callback(self, msg):
        """Handle a full AI response from the legacy topic."""
        self.bridge.send_raw("openchat:1\n")
        self.bridge.send_raw("eyeaction:talk\n")
        payload = f"ai:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'Sent AI text -> {payload.strip()}')

    # ------------------------------------------------------------------
    # tft_cmd: 表情控制指令
    # ------------------------------------------------------------------
    def tft_cmd_callback(self, msg):
        if self.bridge.send_raw(msg.data):
            self.get_logger().debug(f'[Serial] TFT cmd forwarded: {msg.data.strip()}')

    # ------------------------------------------------------------------
    # pca9685_raw: 硬件 15 通道原始值（原 hardware_bridge_node 直接写串口）
    # ------------------------------------------------------------------
    def pca9685_callback(self, msg):
        payload = msg.data + '\n'
        # NETCFG may temporarily own the serial stream. Never block the ROS
        # executor or queue stale servo positions behind that transaction.
        if self.bridge.send_raw(payload, block=False):
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
                    lambda stream: self.netcfg.save_and_apply(settings, stream=stream)
                )
            else:
                data = self.bridge.run_exclusive(lambda stream: self.netcfg.query(stream=stream))
            response.update(ok=True, data=data)
        except (NetworkConfigError, RuntimeError) as exc:
            response["error"] = str(exc)
        except Exception:
            # Do not expose or log request contents; serial details are not useful
            # to the browser and could accidentally include sensitive input.
            response["error"] = "ESP32 网络配置串口通信失败"
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
