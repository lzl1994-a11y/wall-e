#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from ai_msgs.msg import PerceptionTargets
    HAS_MSGS = True
except ImportError:
    HAS_MSGS = False

class AIMSgSyncNode(Node):
    """
    纯粹的时间戳同步节点。
    解决 hobot_codec 硬件转码后时间戳丢失，导致 websocket 无法对齐而报错丢帧的问题。
    """
    def __init__(self):
        super().__init__('ai_msg_sync_node')
        if not HAS_MSGS:
            return
            
        self.latest_raw_stamp = None
        
        # 订阅由第二道 codec 编码出来的 JPEG 获取它最新的时间戳
        self.create_subscription(Image, '/image_padded_jpeg', self.raw_img_cb, 10)
        
        # 订阅原始的 AI 框
        self.create_subscription(PerceptionTargets, '/hobot_mono2d_body_detection_raw', self.ai_cb, 10)
        
        # 发布时间戳同步后的框
        self.pub = self.create_publisher(PerceptionTargets, '/hobot_mono2d_body_detection', 10)
        self.get_logger().info("AI 时间戳同步节点已启动 (纯转发无缩放)")
        
    def raw_img_cb(self, msg):
        self.latest_raw_stamp = msg.header.stamp

    def ai_cb(self, msg):
        # 将 AI 框的时间戳强行篡改为最新相机的帧时间戳
        if self.latest_raw_stamp is not None:
            msg.header.stamp = self.latest_raw_stamp
            
        self.pub.publish(msg)

def main():
    if not HAS_MSGS:
        return
    rclpy.init()
    node = AIMSgSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
