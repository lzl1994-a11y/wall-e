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
        
        # 预先分配一整块连续的 NV12 内存 (Y 占 w*h, UV 占 w*h/2)
        self.nv12_size = self.target_w * self.target_h * 3 // 2
        self.nv12_buffer = np.zeros(self.nv12_size, dtype=np.uint8)
        
        # 使用 numpy 的 view（视图）直接映射到这块内存，避免拷贝
        self.canvas_y = self.nv12_buffer[:self.target_w * self.target_h].reshape((self.target_h, self.target_w))
        self.canvas_uv = self.nv12_buffer[self.target_w * self.target_h:].reshape((self.target_h // 2, self.target_w))
        
        # UV 通道默认填充 128 (纯黑/无色彩)
        self.canvas_uv.fill(128)
        
        # 订阅由 hobot_codec 解码出来的 nv12 图像
        self.create_subscription(Image, '/image_nv12', self.img_cb, 10)
        
        # 发布填充后的 nv12 图像
        self.pub = self.create_publisher(Image, '/image_padded_nv12', 10)
        
        self.get_logger().info("NV12 物理黑边填充节点已启动 (极致性能优化版)")
        
    def img_cb(self, msg):
        try:
            w, h = msg.width, msg.height
            
            if w > self.target_w or h > self.target_h:
                self.get_logger().warn(f"输入图像 ({w}x{h}) 大于画布 ({self.target_w}x{self.target_h})")
                return
                
            # 【核心优化】：绝对不能切片 ROS 的 msg.data！那会导致 Python 底层进行海量内存拷贝
            # 正确做法：一秒钟内将整个 msg.data 零拷贝映射为 numpy 数组，然后在 numpy 里进行 O(1) 瞬间切片
            full_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            y_size = w * h
            y_plane = full_data[:y_size].reshape((h, w))
            uv_plane = full_data[y_size:y_size + (w * h // 2)].reshape((h // 2, w))
            
            # 计算居中坐标
            start_x = (self.target_w - w) // 2
            start_y = (self.target_h - h) // 2
            
            # 直接物理拷贝到预先分配好的视图中 (瞬间完成)
            self.canvas_y[start_y:start_y+h, start_x:start_x+w] = y_plane
            
            start_uv_y = start_y // 2
            self.canvas_uv[start_uv_y:start_uv_y+(h//2), start_x:start_x+w] = uv_plane
            
            # 发布 (直接把连续内存块 tobytes，无需再拼接)
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = self.target_h
            out_msg.width = self.target_w
            out_msg.encoding = 'nv12'
            out_msg.is_bigendian = 0
            out_msg.step = self.target_w
            out_msg.data = self.nv12_buffer.tobytes()
            
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
