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
        
        # 订阅原始 /image (MJPEG) 获取它最新的时间戳
        self.create_subscription(Image, '/image', self.raw_img_cb, 10)
        self.latest_raw_stamp = None
        
        # 订阅原始的偏移框
        self.create_subscription(PerceptionTargets, '/hobot_mono2d_body_detection_raw', self.ai_cb, 10)
        
        # 发布修正后的框
        self.pub = self.create_publisher(PerceptionTargets, '/hobot_mono2d_body_detection', 10)
        self.get_logger().info("AI 坐标系放大校准节点已启动")
        
    def raw_img_cb(self, msg):
        self.latest_raw_stamp = msg.header.stamp

    def img_cb(self, msg):
        if not self.has_resolution or self.img_w != msg.width:
            self.img_w = msg.width
            self.img_h = msg.height
            self.has_resolution = True
            self.get_logger().info(f"检测到真实相机分辨率: {self.img_w}x{self.img_h}")

    def ai_cb(self, msg):
        if not self.has_resolution:
            # 还没有分辨率时，假设默认 1920x1080
            actual_w, actual_h = 1920.0, 1080.0
        else:
            actual_w, actual_h = float(self.img_w), float(self.img_h)
            
        # 核心修复：AI 节点的底层使用了 Letterbox (保持长宽比的边缘填充) 缩放图像
        # 我们必须逆向解算 Letterbox，而不是简单粗暴的拉伸！
        scale = min(self.ai_w / actual_w, self.ai_h / actual_h)
        pad_x = (self.ai_w - actual_w * scale) / 2.0
        pad_y = (self.ai_h - actual_h * scale) / 2.0
            
        # 就地缩放坐标 (逆向去黑边)
        for target in msg.targets:
            for roi in target.rois:
                # 原始中心点或左上角减去 padding 后除以 scale
                new_x = int((roi.rect.x_offset - pad_x) / scale)
                new_y = int((roi.rect.y_offset - pad_y) / scale)
                
                # 宽和高只需要除以 scale，因为它们不受平移(padding)影响
                new_w = int(roi.rect.width / scale)
                new_h = int(roi.rect.height / scale)
                
                # 核心修复 3：防止边缘坐标出现负数导致 uint32 赋值越界崩溃
                roi.rect.x_offset = max(0, new_x)
                roi.rect.y_offset = max(0, new_y)
                roi.rect.width = max(0, new_w)
                roi.rect.height = max(0, new_h)
                
        # 核心修复 2：将 AI 框的时间戳强行篡改为最新相机的帧时间戳
        # 解决 hobot_codec 硬件转码后时间戳丢失，导致 websocket 无法对齐而报错丢帧的问题
        if self.latest_raw_stamp is not None:
            msg.header.stamp = self.latest_raw_stamp
            
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
