#!/usr/bin/env python3
# nodes/doa_ros_node.py
# DOA 声源定位 ROS 桥接节点
# 从串口读取 TDOA 声源定位模块数据，解析角度后发布到 /doa_angle (std_msgs/Int32)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from services.doa_listener import DOAListener


class DoaRosNode(Node):
    """DOA 桥接节点：串口 TDOA -> ROS2 /doa_angle topic"""

    def __init__(self):
        super().__init__('doa_ros_node')

        # 从 ROS 参数读取串口配置，默认值 fallback
        self.declare_parameter('doa_port', '/dev/ttyUSB_DOA')
        self.declare_parameter('doa_baudrate', 115200)

        port = self.get_parameter('doa_port').value
        baudrate = self.get_parameter('doa_baudrate').value

        # 发布角度
        self._pub = self.create_publisher(Int32, '/doa_angle', 10)

        # 启动 DOA 串口监听，收到角度时发布
        self._listener = DOAListener(
            port=port,
            baudrate=baudrate,
            on_angle_received=self._publish_angle
        )

        if self._listener.start():
            self.get_logger().info(f"DOA ROS node started on {port} @ {baudrate}")
        else:
            self.get_logger().error(f"Failed to start DOA listener on {port}")

    def _publish_angle(self, angle):
        """DOA 回调：将角度发布到 /doa_angle"""
        msg = Int32()
        msg.data = angle
        self._pub.publish(msg)
        self.get_logger().info(f"DOA angle published: {angle}°")

    def destroy_node(self):
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