"""手柄测试：发现/监听/断线检测"""
import evdev
import sys
import time
import threading
import os

# ---------------------------------------------------------------------------
# 1. 查找手柄
# ---------------------------------------------------------------------------
device = None
for path in evdev.list_devices():
    d = evdev.InputDevice(path)
    caps = d.capabilities(verbose=False)
    abs_caps = caps.get(3, [])
    key_caps = caps.get(1, [])
    if abs_caps and key_caps:
        print(f"发现手柄: {d.name}")
        print(f"  路径: {d.path}")
        print(f"  phys: {d.phys}")
        print(f"  轴: {len(abs_caps)}  键: {len(key_caps)}")
        device = d
        break

if device is None:
    print("未发现手柄！可用设备:")
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        print(f"  {path}: {d.name}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. 事件监听线程
# ---------------------------------------------------------------------------
AXIS_NAMES = {
    0: "左X", 1: "左Y", 2: "L2", 3: "右X", 4: "右Y", 5: "R2",
    16: "十字X", 17: "十字Y",
}
BTN_NAMES = {
    304: "A(叉)", 305: "B(圈)", 307: "X(方)", 308: "Y(三角)",
    310: "LB", 311: "RB",
    317: "L3", 318: "R3",
    314: "Select", 315: "Start", 316: "PS",
    544: "上", 545: "下", 546: "左", 547: "右",
}

last_event_time = time.time()
running = True
eof_detected = False

def listen():
    global last_event_time, running, eof_detected
    try:
        for event in device.read_loop():
            if not running:
                break
            last_event_time = time.time()

            if event.type == evdev.ecodes.EV_SYN:
                continue

            if event.type == evdev.ecodes.EV_ABS:
                name = AXIS_NAMES.get(event.code, f"轴{event.code}")
                # 归一化
                abs_caps = device.capabilities(verbose=False).get(3, [])
                info = None
                for c, a in abs_caps:
                    if c == event.code:
                        info = a
                        break
                if info:
                    mid = (info.min + info.max) / 2
                    val = (event.value - mid) / (info.max - mid) * 2
                    print(f"  摇杆 {name:6s}  {val:+.2f}")
                else:
                    print(f"  摇杆 {name:6s}  raw={event.value}")

            elif event.type == evdev.ecodes.EV_KEY:
                name = BTN_NAMES.get(event.code, f"按键{event.code}")
                state = "按下" if event.value else "抬起"
                print(f"  按键 {name:12s}  {state}")

    except OSError as e:
        print(f"\n设备断开: {e}")
        eof_detected = True
        running = False

listener = threading.Thread(target=listen, daemon=True)
listener.start()

# ---------------------------------------------------------------------------
# 3. 断线检测：定期尝试读取，OSError = 手柄关闭/拔出
# ---------------------------------------------------------------------------
def check_alive():
    global eof_detected
    while running and not eof_detected:
        idle_sec = time.time() - last_event_time
        print(f"\r[状态] 上次事件: {idle_sec:.0f}秒前 | 手柄设备路径: {device.path} {'存在' if os.path.exists(device.path) else '消失'}", end="", flush=True)
        time.sleep(1)
    print()

checker = threading.Thread(target=check_alive, daemon=True)
checker.start()

print(f"\n开始监听... 按 Ctrl+C 停止")
print("提示: 关闭手柄后观察设备路径是否消失、read_loop 是否抛出 OSError\n")

try:
    listener.join()
except KeyboardInterrupt:
    running = False
    print("\n\n测试结束。")
    device.close()