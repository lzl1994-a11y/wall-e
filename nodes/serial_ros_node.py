#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json  # 🌟 必须引入 json，用来解析动作指令
# 🚀 极其干净地引入下层服务
from services.serial_bridge import SerialBridge

class SerialNode(Node):
    def __init__(self):
        super().__init__('walle_serial_node')
        
        self.get_logger().info('🔌 正在预热串口网桥服务...')
        
        # ==========================================
        # 1. 初始化纯净的底层桥接服务
        # ==========================================
        # 替换 "walle_screen" 为你下位机真实的响应暗号
        self.bridge = SerialBridge(device_name="walle_screen")
        
        if not self.bridge.ser:
            self.get_logger().error('🔴 底层串口桥接失败，请检查硬件连接！')
            # 根据需求，这里可以 return 或者让节点带着错误的躯壳继续跑
            
        self.get_logger().info('✅ 串口通信节点已上线，准备中转消息。')

        # ==========================================
        # 2. 订阅话题并分发给桥接器
        # ==========================================
        # 接收用户说过的话 -> 组装成 you:xxx\n
        self.sub_you = self.create_subscription(
            String, 'corrected_text', self.you_callback, 10)
            
        # 接收大模型的回答 -> 组装成 ai:xxx\n
        self.sub_ai = self.create_subscription(
            String, 'full_ai_text', self.ai_callback, 10)
            
        # 接收动作指令 -> 组装成 action:xxx\n
        self.sub_action = self.create_subscription(
            String, 'action_cmd', self.action_callback, 10)

    def you_callback(self, msg):
        payload = f"you:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'📤 转发用户文本 -> {payload.strip()}')

    def ai_callback(self, msg):
        """处理大模型回复"""
        # 🌟 连招一：在发文字之前，先发一个“正在讲话”的默认眼神指令
        # 这样下位机屏幕不仅会打字，眼睛还会配合眨动或变幻
        self.bridge.send_raw("eyeaction:talk\n")
        
        # 然后再发送完整的聊天文本
        payload = f"ai:{msg.data}\n"
        if self.bridge.send_raw(payload):
            self.get_logger().info(f'📤 转发AI文本与默认眼神 -> {payload.strip()}')
            
    def action_callback(self, msg):
        """处理大模型动作指令 (拦截情绪工具)"""
        try:
            # msg.data 长这样: {"name": "express_emotion", "arguments": "{\"emotion\": \"happy\"}"}
            cmd_data = json.loads(msg.data)
            tool_name = cmd_data.get("name")
            
            # 🌟 连招二：如果发现大模型调用了情绪工具，立刻拦截并翻译为 eyeaction
            if tool_name == "express_emotion":
                # arguments 里面又是一个 JSON 字符串，需要二次解析
                args_str = cmd_data.get("arguments", "{}")
                args = json.loads(args_str)
                emotion = args.get("emotion", "happy") # 取出情绪词
                
                # 组装成下位机认识的指令并发出去
                payload = f"eyeaction:{emotion}\n"
                if self.bridge.send_raw(payload):
                    self.get_logger().info(f'👀 大模型注入灵魂眼神 -> {payload.strip()}')
            
            # 如果是底盘移动等其他动作，保持原样发送
            elif tool_name == "move_chassis":
                payload = f"action:{msg.data}\n"
                if self.bridge.send_raw(payload):
                    self.get_logger().info(f'🦾 转发机械动作指令 -> {payload.strip()}')
                    
            else:
                # 兜底：其他未知指令也原样发出去
                payload = f"action:{msg.data}\n"
                self.bridge.send_raw(payload)
                
        except json.JSONDecodeError:
            self.get_logger().error(f"🔴 动作指令 JSON 解析失败: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"🔴 动作指令处理异常: {e}")

    # ... destroy_node 和 main 函数保持不变 ...

    def destroy_node(self):
        self.get_logger().info('🛑 正在关闭网桥...')
        if hasattr(self, 'bridge'):
            self.bridge.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()