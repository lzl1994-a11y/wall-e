#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

try:
    from ai_msgs.msg import PerceptionTargets
    from sensor_msgs.msg import Image
    HAS_MSGS = True
except ImportError:
    HAS_MSGS = False

class AIMSgScaler(Node):
    """
    终极版：时间戳同步 + Letterbox 逆向坐标缩放 + 安全边界限幅。
    """
    def __init__(self):
        super().__init__('ai_msg_scaler')
        if not HAS_MSGS:
            self.get_logger().error("Missing ai_msgs or sensor_msgs!")
            return
            
        self.img_w = 640
        self.img_h = 480
        self.ai_w = 960.0
        self.ai_h = 544.0
        self.has_resolution = False
        self.latest_raw_stamp = None
        
        # 订阅原始 /image (MJPEG) 获取它的真实宽高和最新时间戳
        self.create_subscription(Image, '/image', self.raw_img_cb, 10)
        
        # 订阅原始的 AI 框
        self.create_subscription(PerceptionTargets, '/hobot_mono2d_body_detection_raw', self.ai_cb, 10)
        
        # 发布修正后的框
        self.pub = self.create_publisher(PerceptionTargets, '/hobot_mono2d_body_detection', 10)
        self.get_logger().info("AI 坐标系校准节点已启动 (包含 Letterbox 逆向恢复)")
        
    def raw_img_cb(self, msg):
        self.latest_raw_stamp = msg.header.stamp
        if not self.has_resolution or self.img_w != msg.width:
            self.img_w = msg.width
            self.img_h = msg.height
            self.has_resolution = True

    def ai_cb(self, msg):
        actual_w, actual_h = float(self.img_w), float(self.img_h)
            
        # 核心修复：底层 AI 模型并没有补黑边，而是直接进行了暴力拉伸！
        # 所以我们只需要做最纯粹的比例逆向缩放即可。
        scale_x = actual_w / self.ai_w
        scale_y = actual_h / self.ai_h
            
        for target in msg.targets:
            for roi in target.rois:
                new_x = int(roi.rect.x_offset * scale_x)
                new_y = int(roi.rect.y_offset * scale_y)
                new_w = int(roi.rect.width * scale_x)
                new_h = int(roi.rect.height * scale_y)
                
                # 核心修复 2：防止越界导致 uint32 崩溃
                roi.rect.x_offset = max(0, new_x)
                roi.rect.y_offset = max(0, new_y)
                roi.rect.width = max(0, new_w)
                roi.rect.height = max(0, new_h)
                
        # 核心修复 3：时间戳强行对齐，解决丢帧卡死
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
