import serial
import serial.tools.list_ports
import time

class SerialBroker:
    """
    瓦力硬件串口发现与仲裁服务
    """
    
    def __init__(self):
        self.device_map = {"walle_doa1": "COM13"}  # 存储设备身份和串口路径的映射，如 {"walle_doa": "/dev/ttyACM1"}

    def scan_and_identify(self):
        """开机点名：遍历所有串口，发送握手暗号"""
        print("🔍 开始硬件全盘扫描...")
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            port_path = port.device
            print(f"  -> 探测物理接口: {port_path}")
            
            try:
                # 用一个比较通用的配置，极短的 timeout 快速试探
                with serial.Serial(port_path, 115200, timeout=1.0) as temp_ser:
                    # 为了兼容 ESP32，试探时也强拉一下 DTR
                    temp_ser.dtr = True
                    temp_ser.rts = True
                    time.sleep(0.5) # 给单片机重启/缓冲的时间
                    temp_ser.reset_input_buffer()
                    
                    # 发送你的暗号 (注意加上换行符，单片机通常靠换行符截断)
                    temp_ser.write(b"getname:WHO_ARE_YOU\n")
                    
                    # 读取单片机的回答
                    response = temp_ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if response.startswith("IAM:"):
                        device_name = response.split(":")[1]
                        self.device_map[device_name] = port_path
                        print(f"     ✅ 认证成功: 发现 '{device_name}' 挂载于 {port_path}")
                    else:
                        print(f"     ❓ 收到未知回复或无回复: {response}")
                        
            except serial.SerialException:
                print(f"     ❌ 接口被占用或无权限")
            except Exception as e:
                print(f"     ⚠️ 探测异常: {e}")

        print("📊 硬件扫描完毕。当前挂载地图:", self.device_map)
        return self.device_map

    def get_port_for(self, device_name):
        """让具体的服务来取自己的串口路径"""
        return self.device_map.get(device_name, None)

# --- 单独测试这个脚本 ---
if __name__ == "__main__":
    broker = SerialBroker()
    broker.scan_and_identify()