import serial
import time
import re
import threading
import sys

class DOAListener:
    """
    瓦力听觉神经封装类 (TDOA 声源定位串口读取服务)
    设计目标: 彻底免疫 Windows CDC 驱动 Bug，暴力抓取，不丢字节
    """
    def __init__(self, port=None, baudrate=115200, on_angle_received=None):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.is_running = False
        self._listen_thread = None
        
        self.on_angle_received = on_angle_received
        self.angle_pattern = re.compile(r'az:(-?\d+)')

    def start(self):
        try:
            # timeout=0.1 是给暴力 read 用的，防止线程死锁
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.ser.dtr = True
            self.ser.rts = True
            time.sleep(0.5) 
            self.ser.reset_input_buffer()
            
            self.is_running = True
            self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._listen_thread.start()
            
            print(f"🟢 瓦力听觉直觉服务已在后台启动 ({self.port})")
            return True
            
        except serial.SerialException as e:
            print(f"🔴 听觉神经初始化失败: {e}")
            return False

    def _listen_loop(self):
        """回归初心的最简逻辑：只用 readline"""
        while self.is_running:
            if self.ser and self.ser.is_open:
                try:
                    # readline() 自身就是最好的缓存区
                    # 它会自动阻塞，直到收到 \n，然后一次性返回完整的一行
                    line_bytes = self.ser.readline()
                    
                    if line_bytes:
                        line = line_bytes.decode('utf-8', errors='ignore').strip()
                        print(line)
                        # 改为匹配新格式：包含 status:alive 和 az:
                        if "status:alive" in line and "az:" in line:
                            match = self.angle_pattern.search(line)
                            if match and self.on_angle_received:
                                self.on_angle_received(int(match.group(1)))
                except Exception:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        if self._listen_thread:
            self._listen_thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("🛑 瓦力听觉直觉服务已安全关闭")

# ==========================================
# 本地测试桩 
# ==========================================
if __name__ == '__main__':
    def handle_new_angle(angle):
        print(f"🎯 [回调触发] 瓦力直觉: 声音来自 {angle}° 方向")

    # 写死刚才跑通的 COM 口
    test_port = 'COM13'  
    
    listener = DOAListener(port=test_port, on_angle_received=handle_new_angle)
    
    if listener.start():
        try:
            print("⏳ 正在监听中... (按 Ctrl+C 退出)")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            listener.stop()