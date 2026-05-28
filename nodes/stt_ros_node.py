#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 引入底层的听觉血肉引擎
from services.stt_service import STTService

class STTNode(Node):
    def __init__(self):
        super().__init__('walle_ear_node')
        
        # 1. 声明发布者：专往 'voice_text' 这个话题里扔字符串
        self.publisher_ = self.create_publisher(String, 'voice_text', 10)
        
        self.get_logger().info('⏳ 正在预热 Sherpa-ONNX 听觉引擎...')
        
        try:
            # 2. 启动底层引擎，并把“发布消息”的动作作为回调函数塞进去
            self.stt_engine = STTService(
                model_dir="F:/well-e-bot/sherpa-onnx", # 注意确认你的实际路径
                on_sentence_received=self.on_speech_detected
            )
            self.stt_engine.start()
            self.get_logger().info('✅ 听觉节点已上线，正在全天候监听环境声音...')
        except Exception as e:
            self.get_logger().error(f'🔴 底层引擎启动失败: {e}')

    def on_speech_detected(self, text):
        """
        传动轴函数：底层一旦断句成功，立刻触发这里
        """
        self.get_logger().info(f'👂 捕捉到人声: "{text}"')
        
        # 将纯文本打包成 ROS 2 的标准 String 消息并广播出去
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)

    def destroy_node(self):
        # 节点被杀死时，务必释放底层麦克风资源，防止端口被占死
        self.get_logger().info('🛑 正在安全关闭麦克风...')
        if hasattr(self, 'stt_engine'):
            self.stt_engine.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()