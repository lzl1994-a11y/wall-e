"""手柄测试：读取 /dev/input/js0 并打印所有事件"""
import evdev
import sys
import time

# 自动查找手柄设备
device = None
for path in evdev.list_devices():
    d = evdev.InputDevice(path)
    caps = d.capabilities(verbose=False)
    abs_caps = caps.get(3, {})
    key_caps = caps.get(1, {})
    # 有 ABS 轴（摇杆）+ KEY 按键 = 手柄/游戏控制器
    if abs_caps and key_caps:
        # 处理 list 类型
        if isinstance(abs_caps, list):
            abs_count = len(abs_caps)
            abs_info = abs_caps[:8]  # 取前几个看看
        else:
            abs_count = len(abs_caps)
            abs_info = abs_caps
        key_count = len(key_caps) if isinstance(key_caps, list) else len(key_caps)
        print(f"发现手柄: {d.name}  [{d.phys}]  路径: {d.path}")
        print(f"  轴数量: {abs_count} 键数量: {key_count}")
        print(f"  轴详情: {abs_info}")
        device = d
        break

if device is None:
    print("未发现手柄设备！")
    print("可用设备:")
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        print(f"  {path}: {d.name}")
    sys.exit(1)

print(f"\n开始监听 {device.name} ...（Ctrl+C 停止）\n")
print(f"{'时间':>12s}  {'类型':>8s}  {'编号':>4s}  {'值':>8s}  {'说明'}")
print("-" * 70)

try:
    for event in device.read_loop():
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp()))

        if event.type == evdev.ecodes.EV_SYN:
            continue  # 同步事件跳过

        # 事件类型
        if event.type == evdev.ecodes.EV_ABS:
            etype = "摇杆轴"
            # 轴名映射
            axis_names = {
                0: "左摇杆X",  1: "左摇杆Y",
                2: "L2扳机",   3: "右摇杆X",
                4: "右摇杆Y",  5: "R2扳机",
                6: "十字键X",  7: "十字键Y",
            }
            desc = axis_names.get(event.code, f"轴{event.code}")
            # 归一化到 -1.0 ~ 1.0
            abs_caps = device.capabilities(verbose=False).get(evdev.ecodes.EV_ABS, [])
            info = None
            for code, ainfo in abs_caps:
                if code == event.code:
                    info = ainfo
                    break
            if info:
                lo, hi = info.min, info.max
                mid = (lo + hi) / 2
                val = (event.value - mid) / (hi - mid) * 2
                desc += f"  ({val:+.2f})"
        elif event.type == evdev.ecodes.EV_KEY:
            etype = "按键"
            btn_names = {
                304: "A", 305: "B", 307: "X", 308: "Y",
                310: "LB", 311: "RB",
                317: "L3(左摇杆按下)", 318: "R3(右摇杆按下)",
                314: "Back", 315: "Start",
                316: "Home",
                544: "十字键↑", 545: "十字键↓",
                546: "十字键←", 547: "十字键→",
            }
            desc = btn_names.get(event.code, f"按键{event.code}")
        else:
            etype = f"类型{event.type}"
            desc = f"code={event.code}"

        print(f"{ts:>12s}  {etype:>8s}  {event.code:>4d}  {event.value:>8d}  {desc}")

except KeyboardInterrupt:
    print("\n\n测试结束。")
    device.close()