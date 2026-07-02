#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image

class NV12PadderNode(Node):
    """
    纯 Numpy NV12 物理黑边填充节点 (无需 cv2, 不会因为缺包崩溃)
    直接将 640x480 的 NV12 图像居中粘贴到 960x544 的黑底画布上。
    """
    def __init__(self):
        super().__init__('nv12_padder_node')
        
        self.target_w = 960
        self.target_h = 544
        
        # NV12 格式：Y 分量 (全分辨率)，UV 分量 (交错，高度减半)
        self.canvas_y = np.zeros((self.target_h, self.target_w), dtype=np.uint8)
        self.canvas_uv = np.full((self.target_h // 2, self.target_w), 128, dtype=np.uint8) # 128 是无色
        
        # 订阅由 hobot_codec 解码出来的 nv12 图像
        self.create_subscription(Image, '/image_nv12', self.img_cb, 10)
        
        # 发布填充后的 nv12 图像
        self.pub = self.create_publisher(Image, '/image_padded_nv12', 10)
        
        self.get_logger().info("NV12 物理黑边填充节点已启动 (纯 Numpy 无依赖)")
        
    def img_cb(self, msg):
        try:
            w, h = msg.width, msg.height
            
            if w > self.target_w or h > self.target_h:
                self.get_logger().warn(f"输入图像 ({w}x{h}) 大于画布 ({self.target_w}x{self.target_h})")
                return
                
            # NV12 数据分离
            y_size = w * h
            y_plane = np.frombuffer(msg.data[:y_size], dtype=np.uint8).reshape((h, w))
            uv_plane = np.frombuffer(msg.data[y_size:], dtype=np.uint8).reshape((h // 2, w))
            
            # 计算居中坐标
            start_x = (self.target_w - w) // 2
            start_y = (self.target_h - h) // 2
            
            # 拷贝 Y 分量 (保持背景黑)
            self.canvas_y[start_y:start_y+h, start_x:start_x+w] = y_plane
            
            # 拷贝 UV 分量 (高度要除以 2)
            start_uv_y = start_y // 2
            self.canvas_uv[start_uv_y:start_uv_y+(h//2), start_x:start_x+w] = uv_plane
            
            # 合并为最终的 NV12 数据
            nv12_data = np.concatenate((self.canvas_y.flatten(), self.canvas_uv.flatten()))
            
            # 发布
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = self.target_h
            out_msg.width = self.target_w
            out_msg.encoding = 'nv12' # TROS 可识别
            out_msg.is_bigendian = 0
            out_msg.step = self.target_w
            out_msg.data = nv12_data.tobytes()
            
            self.pub.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"NV12 填充失败: {e}")

def main():
    rclpy.init()
    node = NV12PadderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
