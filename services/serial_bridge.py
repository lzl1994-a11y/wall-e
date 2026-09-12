# services/serial_bridge.py
import serial
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from services.serial_broker import SerialBroker
from services.usb_devices import DEFAULT_CONFIG_PATH, serial_ports_for_role


RECONNECT_INITIAL_DELAY_SEC = 1.0
RECONNECT_MAX_DELAY_SEC = 30.0


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
        self._reconnect_delay_sec = RECONNECT_INITIAL_DELAY_SEC
        self._next_selection_check_at = 0.0
        self._selection_config_mtime_ns = self._config_mtime_ns()
        # Only physical writes are serialized. A permanent reader dispatches
        # NETCFG replies by sequence, so waiting for a long Wi-Fi/TCP APPLY does
        # not block screen or motion writes.
        self._io_lock = threading.RLock()
        self._connection_lock = threading.RLock()
        self._response_condition = threading.Condition()
        self._netcfg_responses = defaultdict(deque)
        self._registered_netcfg_sequences = set()
        self._reader_stop = threading.Event()
        self._reader_thread = None
        
        # 🌟 新增：状态机与时间戳管理
        self.last_send_time = 0.0      # 上次成功发送数据的时间戳
        self.is_screen_awake = False   # 屏幕是否处于聊天页面状态
        
        self._ensure_connected()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="esp32-serial-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _connect(self):
        """内部方法：通过 Broker 获取端口并连接"""
        with self._connection_lock:
            if self.ser and self.ser.is_open:
                return
            print(f"🔌 [Serial Bridge] 正在请求挂载设备: {self.device_name}...")
            self.broker.scan_and_identify(
                usb_role="screen_motion",
                fallback_device_name=self.device_name,
            )
            port_path = self.broker.get_port_for(self.device_name)

            if port_path:
                try:
                    self.ser = serial.Serial(
                        port_path,
                        baudrate=115200,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=0.1,
                        write_timeout=2,
                    )
                    print(f"✅ [Serial Bridge] 成功连接下位机: {port_path}")
                    return True
                except Exception as e:
                    print(f"🔴 [Serial Bridge] 串口被占用或无权限: {e}")
                    self.ser = None
            else:
                print(f"🔴 [Serial Bridge] 未能在物理总线上找到设备 '{self.device_name}'")
                self.ser = None
            return False

    def _config_mtime_ns(self):
        try:
            return Path(self.broker.config_path).stat().st_mtime_ns
        except OSError:
            return None

    def _ensure_connected(self):
        with self._connection_lock:
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
            if self._connect():
                self._next_reconnect_at = 0.0
                self._reconnect_delay_sec = RECONNECT_INITIAL_DELAY_SEC
                return True
            self._next_reconnect_at = now + self._reconnect_delay_sec
            self._reconnect_delay_sec = min(
                RECONNECT_MAX_DELAY_SEC,
                self._reconnect_delay_sec * 2.0,
            )
            return False

    @staticmethod
    def _netcfg_sequence(line):
        for prefix in ("NETCFG:RESULT:", "NETCFG:STATUS:"):
            if line.startswith(prefix):
                try:
                    value = int(line[len(prefix):].split("|", 1)[0])
                except (ValueError, IndexError):
                    return None
                return value if 0 <= value <= 0xFFFFFFFF else None
        return None

    def _reader_loop(self):
        """Be the sole physical reader and route correlated NETCFG lines."""
        buffer = bytearray()
        active_stream = None
        while not self._reader_stop.is_set():
            if not self._ensure_connected():
                self._reader_stop.wait(0.1)
                continue
            stream = self.ser
            if stream is not active_stream:
                buffer.clear()
                active_stream = stream
            try:
                chunk = stream.read(1)
            except Exception:
                self._mark_disconnected(stream)
                continue
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "ignore")
            for byte in chunk:
                if byte not in (10, 13):
                    if len(buffer) < 4096:
                        buffer.append(byte)
                    continue
                if not buffer:
                    continue
                line = buffer.decode("utf-8", errors="replace")
                buffer.clear()
                sequence = self._netcfg_sequence(line)
                if sequence is None:
                    continue
                with self._response_condition:
                    if sequence in self._registered_netcfg_sequences:
                        self._netcfg_responses[sequence].append(line)
                        self._response_condition.notify_all()

    def _mark_disconnected(self, stream):
        with self._connection_lock:
            if self.ser is not stream:
                return
            try:
                stream.close()
            except Exception:
                pass
            self.ser = None
            self.is_screen_awake = False

    def _register_netcfg_sequence(self, sequence):
        with self._response_condition:
            self._registered_netcfg_sequences.add(sequence)

    def _unregister_netcfg_sequences(self, sequences):
        with self._response_condition:
            for sequence in sequences:
                self._registered_netcfg_sequences.discard(sequence)
                self._netcfg_responses.pop(sequence, None)

    def _wait_netcfg_line(self, sequences, timeout=0.1):
        deadline = time.monotonic() + timeout
        with self._response_condition:
            while True:
                for sequence in sequences:
                    queue = self._netcfg_responses.get(sequence)
                    if queue:
                        return queue.popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._reader_stop.is_set():
                    return None
                self._response_condition.wait(remaining)

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

    def send_raw(self, payload: str, *, block=True, wake_screen=True):
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
                # A persistent TFT surface (such as music) owns navigation.
                # Motion and eye commands must not replace it with the chat page.
                wake_cmd = self._check_and_wake_screen() if wake_screen else ""
                self.ser.write((wake_cmd + payload).encode("gbk"))
                self.last_send_time = current_time
                return True
            except Exception as exc:
                print(f"⚠️ [Serial Bridge] 发送失败: {exc}")
                self._mark_disconnected(self.ser)
                return False
        finally:
            self._io_lock.release()

    def run_exclusive(self, operation):
        """Run a routed NETCFG transaction without monopolizing normal writes."""
        if not self._ensure_connected():
            raise RuntimeError("ESP32 USB 串口未连接")
        session = _RoutedNetcfgStream(self)
        try:
            return operation(session)
        finally:
            session.close()

    def close(self):
        """安全释放串口"""
        if hasattr(self, "_reader_stop"):
            self._reader_stop.set()
            with self._response_condition:
                self._response_condition.notify_all()
        reader = getattr(self, "_reader_thread", None)
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        with self._io_lock:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("🛑 [Serial Bridge] 串口已安全释放")


class _RoutedNetcfgStream:
    """Serial-like view backed by the bridge's permanent reader."""

    def __init__(self, bridge):
        self._bridge = bridge
        self._sequences = set()
        self._read_buffer = bytearray()

    def write(self, payload):
        raw = bytes(payload)
        try:
            text = raw.decode("ascii").strip()
            sequence = int(text.split(":", 2)[2].split("|", 1)[0])
        except (UnicodeDecodeError, ValueError, IndexError) as exc:
            raise RuntimeError("无效的 NETCFG 串口命令") from exc
        self._sequences.add(sequence)
        self._bridge._register_netcfg_sequence(sequence)
        if not self._bridge.send_raw(text + "\r\n", wake_screen=False):
            raise RuntimeError("ESP32 USB 串口写入失败")
        return len(raw)

    def flush(self):
        return None

    def read(self, count=1):
        if not self._read_buffer:
            line = self._bridge._wait_netcfg_line(self._sequences, timeout=0.05)
            if line is None:
                return b""
            self._read_buffer.extend(line.encode("utf-8") + b"\n")
        result = bytes(self._read_buffer[:count])
        del self._read_buffer[:count]
        return result

    def close(self):
        self._bridge._unregister_netcfg_sequences(self._sequences)
