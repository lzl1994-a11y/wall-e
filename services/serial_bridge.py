# services/serial_bridge.py
import serial
import time
from services.serial_broker import SerialBroker

class SerialBridge:
    """
    瓦力纯净硬件网桥服务 (完全解耦 ROS)
    职责：连接下位机，提供最基础的发送接口，并自动管理屏幕的唤醒状态。
    """
    def __init__(self, device_name="", timeout_seconds=30.0):
        self.device_name = device_name
        self.timeout_seconds = timeout_seconds
        
        self.ser = None
        self.broker = SerialBroker()
        
        # 🌟 新增：状态机与时间戳管理
        self.last_send_time = 0.0      # 上次成功发送数据的时间戳
        self.is_screen_awake = False   # 屏幕是否处于聊天页面状态
        
        self._connect()

    def _connect(self):
        """内部方法：通过 Broker 获取端口并连接"""
        print(f"🔌 [Serial Bridge] 正在请求挂载设备: {self.device_name}...")
        self.broker.scan_and_identify()
        port_path = self.broker.get_port_for(self.device_name)
        
        if port_path:
            try:
                self.ser = serial.Serial(port_path, 115200, timeout=1.0)
                print(f"✅ [Serial Bridge] 成功连接下位机: {port_path}")
            except Exception as e:
                print(f"🔴 [Serial Bridge] 串口被占用或无权限: {e}")
                self.ser = None
        else:
            print(f"🔴 [Serial Bridge] 未能在物理总线上找到设备 '{self.device_name}'")
            self.ser = None

    def _check_and_wake_screen(self):
        """
        核心拦截器：检查是否需要唤醒屏幕。
        如果距离上次发送超过 30 秒，或者屏幕从未被唤醒过，则返回唤醒指令字符串；否则返回空字符串。
        """
        current_time = time.time()
        
        # 如果是第一次，或者超时了 30 秒
        if not self.is_screen_awake or (current_time - self.last_send_time > self.timeout_seconds):
            print("📺 [Serial Bridge] 屏幕休眠中或首次对话，注入唤醒指令 (openchat:1)")
            self.is_screen_awake = True
            return "openchat:1\n"
        
        return ""

    def send_raw(self, payload: str):
        """
        核心发送方法：外部只需传入组装好的字符串 (如 'you:xxx\\n')。
        内部会自动判断是否需要先发送唤醒指令。
        """
        if self.ser and self.ser.is_open:
            try:
                current_time = time.time()
                
                # 1. 获取可能需要的唤醒指令前缀
                wake_cmd = self._check_and_wake_screen()
                
                # 2. 将唤醒指令和真实的 payload 拼接在一起
                # 例如："openchat:1\nyou:你好\n" 或者单纯的 "you:你好\n"
                final_payload = wake_cmd + payload
                
                # 3. 发送给下位机 (注意编码格式)
                self.ser.write(final_payload.encode('gbk'))
                
                # 4. 刷新最后发送的时间戳
                self.last_send_time = current_time
                
                return True
            except Exception as e:
                print(f"⚠️ [Serial Bridge] 发送失败: {e}")
                # 发送失败的话，认为屏幕可能没收到，重置唤醒状态
                self.is_screen_awake = False 
                return False
        else:
            print("⚠️ [Serial Bridge] 串口未连接，指令丢弃。")
            self.is_screen_awake = False
            return False

    def close(self):
        """安全释放串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("🛑 [Serial Bridge] 串口已安全释放")