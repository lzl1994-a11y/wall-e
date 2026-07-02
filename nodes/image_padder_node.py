#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

try:
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_CV = True
except ImportError:
    HAS_CV = False

class ImagePadderNode(Node):
    """
    终极物理对齐节点：
    直接接收 640x480 的画面，并在周围补黑边，强行凑成 960x544 的画布。
    这样一来，AI 模型和 Websocket 看到的都是 960x544 的物理画面，
    任何坐标都不需要再转换，百分之百 1:1 绝对对齐。
    """
    def __init__(self):
        super().__init__('image_padder_node')
        if not HAS_CV:
            self.get_logger().error("Missing cv_bridge or sensor_msgs!")
            return
            
        self.bridge = CvBridge()
        
        self.target_w = 960
        self.target_h = 544
        
        # 预先分配一块黑色的画布，避免每次分配内存
        self.canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        
        self.create_subscription(Image, '/image_raw', self.img_cb, 10)
        self.pub_bgr = self.create_publisher(Image, '/image_padded_bgr', 10)
        self.pub_jpeg = self.create_publisher(Image, '/image_padded_jpeg', 10)
        
        self.get_logger().info("图像物理黑边填充节点已启动 (输出 BGR 和 JPEG)")
        
    def img_cb(self, msg):
        try:
            # 假设输入是 bgr8
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = cv_img.shape[:2]
            
            # 计算居中坐标
            start_x = (self.target_w - w) // 2
            start_y = (self.target_h - h) // 2
            
            if start_x < 0 or start_y < 0:
                self.get_logger().warn("输入图像过大")
                return
                
            # 物理拷贝到画布中心
            self.canvas[start_y:start_y+h, start_x:start_x+w] = cv_img
            
            # 1. 发布 BGR 给地平线 AI 节点
            out_bgr = self.bridge.cv2_to_imgmsg(self.canvas, encoding='bgr8')
            out_bgr.header = msg.header
            self.pub_bgr.publish(out_bgr)
            
            # 2. 发布 JPEG 给 Websocket 网页
            import cv2
            _, jpeg_data = cv2.imencode('.jpg', self.canvas)
            out_jpeg = Image()
            out_jpeg.header = msg.header
            out_jpeg.height = self.target_h
            out_jpeg.width = self.target_w
            out_jpeg.encoding = 'jpeg'
            out_jpeg.is_bigendian = 0
            out_jpeg.step = self.target_w * 3
            out_jpeg.data = jpeg_data.tobytes()
            self.pub_jpeg.publish(out_jpeg)
            
        except Exception as e:
            self.get_logger().error(f"填充图像失败: {e}")

def main():
    if not HAS_CV:
        return
    rclpy.init()
    node = ImagePadderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
