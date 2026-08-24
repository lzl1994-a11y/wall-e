# services/serial_bridge.py
import serial
import threading
import time
from pathlib import Path
from services.serial_broker import SerialBroker
from services.usb_devices import DEFAULT_CONFIG_PATH, serial_ports_for_role

class SerialBridge:
    """
    瓦力纯净硬件网桥服务 (完全解耦 ROS)
    职责：连接下位机，提供最基础的发送接口，并自动管理屏幕的唤醒状态。
    """
    def __init__(self, device_name="WALL_E_TFT", timeout_seconds=30.0, config_path=DEFAULT_CONFIG_PATH):
        self.device_name = device_name
        self.timeout_seconds = timeout_seconds
        
        self.ser = None
        self.broker = SerialBroker(config_path=config_path)
        self._next_reconnect_at = 0.0
        self._next_selection_check_at = 0.0
        self._selection_config_mtime_ns = self._config_mtime_ns()
        # All serial reads/writes, including long NETCFG APPLY transactions, use
        # this one lock. This is the sole holder of the ESP32 USB device.
        self._io_lock = threading.RLock()
        
        # 🌟 新增：状态机与时间戳管理
        self.last_send_time = 0.0      # 上次成功发送数据的时间戳
        self.is_screen_awake = False   # 屏幕是否处于聊天页面状态
        
        self._connect()

    def _connect(self):
        """内部方法：通过 Broker 获取端口并连接"""
        print(f"🔌 [Serial Bridge] 正在请求挂载设备: {self.device_name}...")
        self.broker.scan_and_identify(usb_role="screen_motion")
        port_path = self.broker.get_port_for(self.device_name)
        
        if port_path:
            try:
                self.ser = serial.Serial(
                    port_path,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1.0,
                    write_timeout=2,
                )
                print(f"✅ [Serial Bridge] 成功连接下位机: {port_path}")
            except Exception as e:
                print(f"🔴 [Serial Bridge] 串口被占用或无权限: {e}")
                self.ser = None
        else:
            print(f"🔴 [Serial Bridge] 未能在物理总线上找到设备 '{self.device_name}'")
            self.ser = None

    def _config_mtime_ns(self):
        try:
            return Path(self.broker.config_path).stat().st_mtime_ns
        except OSError:
            return None

    def _ensure_connected(self):
        now = time.monotonic()
        if self.ser and self.ser.is_open and now >= self._next_selection_check_at:
            self._next_selection_check_at = now + 1.0
            config_mtime_ns = self._config_mtime_ns()
            if config_mtime_ns != self._selection_config_mtime_ns:
                self._selection_config_mtime_ns = config_mtime_ns
                selected_ports, configured = serial_ports_for_role(
                    "screen_motion", self.broker.config_path
                )
                if configured and self.ser.port not in selected_ports:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
        if self.ser and self.ser.is_open:
            return True
        if now < self._next_reconnect_at:
            return False
        self._next_reconnect_at = now + 1.0
        self._connect()
        return bool(self.ser and self.ser.is_open)

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

    def send_raw(self, payload: str, *, block=True):
        """Send normal screen/motion traffic while holding the shared USB lock."""
        if not self._io_lock.acquire(blocking=block):
            return False
        try:
            if not self._ensure_connected():
                print("⚠️ [Serial Bridge] 串口未连接，指令丢弃。")
                self.is_screen_awake = False
                return False
            try:
                current_time = time.time()
                wake_cmd = self._check_and_wake_screen()
                self.ser.write((wake_cmd + payload).encode("gbk"))
                self.last_send_time = current_time
                return True
            except Exception as exc:
                print(f"⚠️ [Serial Bridge] 发送失败: {exc}")
                self.is_screen_awake = False
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return False
        finally:
            self._io_lock.release()

    def run_exclusive(self, operation):
        """Run a serial transaction without injecting screen wake/display commands.

        NETCFG has request/response framing and must not be interleaved with the
        normal screen/motion traffic.  The callback receives the already-open
        pyserial object and must not close it.
        """
        with self._io_lock:
            if not self._ensure_connected():
                raise RuntimeError("ESP32 USB 串口未连接")
            try:
                return operation(self.ser)
            except Exception:
                # Keep reconnect semantics consistent with normal send failures.
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self.is_screen_awake = False
                raise

    def close(self):
        """安全释放串口"""
        with self._io_lock:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("🛑 [Serial Bridge] 串口已安全释放")
