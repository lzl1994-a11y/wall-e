import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class YoloBrainNode(Node):
    def __init__(self):
        # 1. 给这个节点起个名字，它就像连接神经系统的网卡 MAC 地址
        super().__init__('yolo_brain_node')
        
        # 2. 创建一个发布者。往 '/wall_e/vision' 这个话题里发送字符串消息
        self.publisher = self.create_publisher(String, '/wall_e/vision', 10)
        
        # 3. 创建一个定时器，每 1 秒钟执行一次 timer_callback 函数（模拟摄像头帧率）
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # 4. 把要发送的数据打包
        msg = String()
        msg.data = '瓦力视觉：前方发现红色小球！目标坐标 X=150, Y=200'
        
        # 5. 通过神经系统广播出去
        self.publisher.publish(msg)
        
        # 6. 在自己的终端打印日志，方便你看到运行状态
        self.get_logger().info(f'正在广播: {msg.data}')

def main(args=None):
    # 初始化 ROS 神经系统
    rclpy.init(args=args)
    # 实例化你的视觉大脑节点
    node = YoloBrainNode()
    # 让节点持续运行，不要退出
    rclpy.spin(node)
    
    # 退出时的清理工作
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
