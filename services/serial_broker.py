import serial
import serial.tools.list_ports
import time

from services.usb_devices import DEFAULT_CONFIG_PATH, serial_ports_for_role

class SerialBroker:
    """
    瓦力硬件串口发现与仲裁服务
    """
    
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.device_map = {}

    def scan_and_identify(self, usb_role=None, fallback_device_name=None):
        """Probe serial ports and identify devices by their application handshake.

        A configured USB selector is a preferred route.  For callers that pass
        ``fallback_device_name``, a stale selector may fall back to the other
        serial ports, but only the requested handshake identity is accepted.
        Other roles retain strict selector behavior so, for example, DOA
        discovery cannot reset the screen controller.
        """
        print("🔍 开始硬件全盘扫描...")
        all_ports = list(serial.tools.list_ports.comports())
        ports = all_ports
        fallback_ports = []
        configured = False
        if usb_role:
            selected_ports, configured = serial_ports_for_role(usb_role, self.config_path)
            if configured:
                selected_paths = set(selected_ports)
                ports = [port for port in all_ports if port.device in selected_paths]
                fallback_ports = [
                    port for port in all_ports if port.device not in selected_paths
                ]
                if not ports:
                    print(
                        f"  -> 已配置的 {usb_role} USB 当前离线或没有串口接口"
                    )
        self.device_map = {}

        self._probe_ports(ports)
        if (
            configured
            and fallback_device_name
            and fallback_device_name not in self.device_map
            and fallback_ports
        ):
            print(
                f"  -> USB 选择器未找到 '{fallback_device_name}'，"
                "回退到串口握手发现"
            )
            self._probe_ports(fallback_ports)

        print("📊 硬件扫描完毕。当前挂载地图:", self.device_map)
        return self.device_map

    def _probe_ports(self, ports):
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

    def get_port_for(self, device_name):
        """让具体的服务来取自己的串口路径"""
        return self.device_map.get(device_name, None)

# --- 单独测试这个脚本 ---
if __name__ == "__main__":
    broker = SerialBroker()
    broker.scan_and_identify()
