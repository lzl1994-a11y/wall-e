"""Read direction-of-arrival angles from the dedicated serial sensor.

English: ``DOAListener`` owns only the serial connection and background read
thread.  A valid firmware heartbeat line containing ``status:alive`` and an
``az:<degrees>`` field is converted to an integer callback.  ROS publication,
tracking decisions, and head motion remain the caller's responsibility.

中文：``DOAListener`` 只管理声源定位传感器的串口连接和后台读取线程。固件必须发送同时
包含 ``status:alive`` 与 ``az:<角度>`` 的完整行，服务才会把角度转换成整数并触发回调。
ROS 发布、目标跟踪和头部运动决策均由调用方负责，避免传感器驱动包含业务行为。
"""

import serial
import time
import re
import threading
import sys

class DOAListener:
    """Manage one TDOA serial reader and deliver validated azimuth callbacks.

    The reader uses a short serial timeout so ``stop`` can terminate promptly.
    Input decoding is tolerant of invalid UTF-8 bytes, but a callback is emitted
    only for a complete alive/status line matching the documented angle format.

    管理一个 TDOA 串口读取器，并只对校验通过的方位角触发回调。较短的串口超时保证
    ``stop`` 不会被永久阻塞；无效 UTF-8 字节会被忽略，但不完整或不含存活状态的行不会
    被当作有效测量，从而避免串口噪声触发运动。
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
        """Open the configured port and start the daemon reader thread.

        打开配置串口并启动守护读取线程。初始化失败时返回 ``False``，不会留下一个标记为
        运行中但实际没有串口的半初始化实例；成功后返回 ``True``。
        """
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
                    self.is_running = False
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
                    break
            else:
                time.sleep(0.1)

    def stop(self):
        """Stop reading, join a foreign reader thread, and close the port.

        停止读取；若由其他线程调用则等待后台线程退出，随后关闭串口。读取线程内部调用
        清理时不会等待自身，避免发生自连接死锁。
        """
        self.is_running = False
        if self._listen_thread and self._listen_thread is not threading.current_thread():
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
