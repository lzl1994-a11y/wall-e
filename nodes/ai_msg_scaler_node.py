#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

try:
    from ai_msgs.msg import PerceptionTargets
    from sensor_msgs.msg import Image
    HAS_MSGS = True
except ImportError:
    HAS_MSGS = False

class AIMSgScaler(Node):
    """
    修正 AI 框因为摄像头分辨率与模型分辨率不匹配导致的 Web 页面漂移问题。
    动态读取实际的视频流分辨率，并对 960x544 坐标系下的 AI 框进行等比放大。
    """
    def __init__(self):
        super().__init__('ai_msg_scaler')
        if not HAS_MSGS:
            self.get_logger().error("Missing ai_msgs or sensor_msgs!")
            return
            
        self.img_w = 1920
        self.img_h = 1080
        self.ai_w = 960.0
        self.ai_h = 544.0
        self.has_resolution = False
        
        # 订阅 NV12 图像仅为了获取真实的宽高
        self.create_subscription(Image, '/image_nv12', self.img_cb, 10)
        
        # 订阅原始的偏移框
        self.create_subscription(PerceptionTargets, '/hobot_mono2d_body_detection_raw', self.ai_cb, 10)
        
        # 发布修正后的框
        self.pub = self.create_publisher(PerceptionTargets, '/hobot_mono2d_body_detection', 10)
        self.get_logger().info("AI 坐标系放大校准节点已启动")
        
    def img_cb(self, msg):
        if not self.has_resolution or self.img_w != msg.width:
            self.img_w = msg.width
            self.img_h = msg.height
            self.has_resolution = True
            self.get_logger().info(f"检测到真实相机分辨率: {self.img_w}x{self.img_h}")

    def ai_cb(self, msg):
        if not self.has_resolution:
            # 还没有分辨率时，假设默认 1920x1080
            scale_x = 1920.0 / self.ai_w
            scale_y = 1080.0 / self.ai_h
        else:
            scale_x = self.img_w / self.ai_w
            scale_y = self.img_h / self.ai_h
            
        # 就地缩放坐标
        for target in msg.targets:
            for roi in target.rois:
                roi.rect.x_offset = int(roi.rect.x_offset * scale_x)
                roi.rect.y_offset = int(roi.rect.y_offset * scale_y)
                roi.rect.width = int(roi.rect.width * scale_x)
                roi.rect.height = int(roi.rect.height * scale_y)
                
        self.pub.publish(msg)

def main():
    if not HAS_MSGS:
        return
    rclpy.init()
    node = AIMSgScaler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
