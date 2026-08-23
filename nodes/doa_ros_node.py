#!/usr/bin/env python3
# nodes/doa_ros_node.py
# DOA 声源定位 ROS 桥接节点
# 通过 SerialBroker 自动发现 DOA 模块串口 → 解析角度 → /doa_angle topic

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from services.serial_broker import SerialBroker
from services.doa_listener import DOAListener
from services.usb_devices import serial_ports_for_role


class DoaRosNode(Node):
    """DOA 桥接节点：串口自动发现 → TDOA 角度 → ROS2 /doa_angle"""

    def __init__(self):
        super().__init__('doa_ros_node')

        # 发布角度
        self._pub = self.create_publisher(Int32, '/doa_angle', 10)
        self._broker = SerialBroker()
        self._listener = None
        self._last_wait_log = 0.0
        self._reconnect_timer = self.create_timer(1.0, self._ensure_listener)
        self._ensure_listener()

    def _ensure_listener(self):
        selected_ports, configured = serial_ports_for_role("voice")
        # Audio capture/playback can safely use the system default device, but
        # DOA is a serial protocol.  Falling back to probing every ACM/USB port
        # here also probes WALL_E_TFT, resets its serial buffers and races the
        # screen/motion bridge.  Only start DOA discovery after the user has
        # explicitly assigned a USB device to the voice role.
        if not configured:
            if self._listener:
                self._listener.stop()
                self._listener = None
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_wait_log >= 10.0:
                self.get_logger().info("未配置 voice USB，跳过 ESP_MIC 串口探测")
                self._last_wait_log = now
            return

        if self._listener and self._listener.is_running:
            if self._listener.port in selected_ports:
                return
            self.get_logger().info("语音 USB 配置已变化，重新连接 DOA")
            self._listener.stop()
            self._listener = None
        if self._listener:
            self._listener.stop()
            self._listener = None

        self._broker.scan_and_identify(usb_role="voice")
        port = self._broker.get_port_for("ESP_MIC")
        if not port:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_wait_log >= 10.0:
                self.get_logger().warn("等待语音 USB / ESP_MIC 串口接入")
                self._last_wait_log = now
            return

        listener = DOAListener(port=port, baudrate=115200, on_angle_received=self._publish_angle)
        if listener.start():
            self._listener = listener
            self.get_logger().info(f"DOA 已连接 {port}，发布 /doa_angle")
        else:
            listener.stop()

    def _publish_angle(self, angle):
        """DOA 回调：将角度发布到 /doa_angle"""
        msg = Int32()
        msg.data = angle
        self._pub.publish(msg)
        self.get_logger().debug(f"DOA angle published: {angle}°")

    def destroy_node(self):
        if self._listener:
            self._listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DoaRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
