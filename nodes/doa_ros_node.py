#!/usr/bin/env python3
# nodes/doa_ros_node.py
# DOA 声源定位 ROS 桥接节点
# 通过 SerialBroker 自动发现 DOA 模块串口 → 解析角度 → /doa_angle topic

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from services.serial_broker import SerialBroker
from services.doa_listener import DOAListener


class DoaRosNode(Node):
    """DOA 桥接节点：串口自动发现 → TDOA 角度 → ROS2 /doa_angle"""

    def __init__(self):
        super().__init__('doa_ros_node')

        # 通过 SerialBroker 自动发现 DOA 模块串口
        broker = SerialBroker()
        broker.scan_and_identify()
        port = broker.get_port_for("IAM:ESP_MIC")

        if not port:
            self.get_logger().error(
                "SerialBroker 未发现 DOA 模块 (IAM:ESP_MIC)，"
                "检查硬件连接和握手固件"
            )
            self._listener = None
            return

        self.get_logger().info(f"DOA 模块发现于 {port}")

        # 发布角度
        self._pub = self.create_publisher(Int32, '/doa_angle', 10)

        # 启动 DOA 串口监听
        self._listener = DOAListener(
            port=port,
            baudrate=115200,
            on_angle_received=self._publish_angle
        )

        if self._listener.start():
            self.get_logger().info(f"DOA ROS node started, publishing to /doa_angle")
        else:
            self.get_logger().error(f"DOA listener 启动失败 ({port})")
            self._listener = None

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
    if node._listener is None:
        node.get_logger().error("DOA node 因无设备退出")
        node.destroy_node()
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()