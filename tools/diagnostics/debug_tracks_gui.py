#!/usr/bin/env python3
"""A small desktop control panel for safely debugging Wall-E's tracks.

Start it with ``python tools/diagnostics/debug_tracks_gui.py``.  It uses the
same serial packet format and configuration as debug_tracks.py, and never
changes the configured servo initial positions.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import serial
import serial.tools.list_ports

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnostics.debug_tracks import (
    encode_packet,
    load_hardware_config,
    motion_state,
    resolve_port,
    send_for_duration,
)


class TrackDebugApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wall-E 履带串口调试")
        self.resizable(False, False)
        self.config_path = ROOT / "core" / "config.yaml"
        self.config_data = load_hardware_config(self.config_path)
        self.stream = None
        self.io_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.busy = False

        self.port = tk.StringVar()
        self.speed = tk.IntVar(value=20)
        self.duration = tk.DoubleVar(value=0.5)
        self.status = tk.StringVar(value="未连接")
        self._build()
        self.refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        outer = ttk.Frame(self, padding=16)
        outer.grid()
        ttk.Label(outer, text="串口").grid(row=0, column=0, sticky="w")
        self.port_box = ttk.Combobox(outer, textvariable=self.port, width=23)
        self.port_box.grid(row=0, column=1, padx=8)
        ttk.Button(outer, text="刷新", command=self.refresh_ports).grid(row=0, column=2)
        self.connect_button = ttk.Button(outer, text="连接", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=3, padx=(8, 0))

        ttk.Label(outer, text="速度 (%)").grid(row=1, column=0, sticky="w", pady=(14, 0))
        ttk.Scale(outer, from_=1, to=100, variable=self.speed, orient="horizontal", length=220).grid(
            row=1, column=1, columnspan=2, sticky="we", pady=(14, 0)
        )
        ttk.Label(outer, textvariable=self.speed, width=4).grid(row=1, column=3, pady=(14, 0))

        ttk.Label(outer, text="单次时长 (秒)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(outer, from_=0.1, to=10.0, increment=0.1, textvariable=self.duration, width=8).grid(
            row=2, column=1, sticky="w", padx=8, pady=(10, 0)
        )
        ttk.Label(outer, text="每次动作自动停车", foreground="#555").grid(
            row=2, column=2, columnspan=2, sticky="w", pady=(10, 0)
        )

        controls = ttk.LabelFrame(outer, text="履带控制", padding=12)
        controls.grid(row=3, column=0, columnspan=4, pady=(16, 8), sticky="we")
        self.motion_buttons = []
        layout = (("前进 ▲", "forward", 0, 1), ("左转 ◀", "left", 1, 0),
                  ("后退 ▼", "backward", 1, 1), ("右转 ▶", "right", 1, 2))
        for label, motion, row, column in layout:
            button = ttk.Button(controls, text=label, command=lambda m=motion: self.start_motion(m), width=12)
            button.grid(row=row, column=column, padx=5, pady=5)
            self.motion_buttons.append(button)
        self.stop_button = tk.Button(
            controls, text="紧急停车", command=self.emergency_stop, bg="#c62828", fg="white",
            activebackground="#8e0000", activeforeground="white", width=12,
        )
        self.stop_button.grid(row=2, column=0, columnspan=3, padx=5, pady=(8, 2))

        ttk.Separator(outer).grid(row=4, column=0, columnspan=4, sticky="we", pady=8)
        ttk.Label(outer, textvariable=self.status, foreground="#1b5e20").grid(row=5, column=0, columnspan=4, sticky="w")
        ttk.Label(outer, text="舵机通道 0–8 始终使用 config.yaml 的 init 值。", foreground="#555").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(5, 0)
        )

    def refresh_ports(self):
        ports = [item.device for item in serial.tools.list_ports.comports()]
        self.port_box["values"] = ports
        if not self.port.get():
            try:
                self.port.set(resolve_port(self.config_data, None, self.config_path))
            except RuntimeError:
                if ports:
                    self.port.set(ports[0])

    def toggle_connection(self):
        if self.stream and self.stream.is_open:
            self.close_serial()
            return
        port = self.port.get().strip()
        if not port:
            messagebox.showerror("未选择串口", "请选择 ESP32 的串口后再连接。")
            return
        try:
            # Do not toggle DTR/RTS: that may reset an ESP32 connected by USB.
            self.stream = serial.Serial(port, baudrate=115200, timeout=1, write_timeout=2)
        except (OSError, serial.SerialException) as exc:
            messagebox.showerror("连接失败", str(exc))
            self.stream = None
            return
        self.connect_button.configure(text="断开")
        self.status.set(f"已连接 {port}；可以开始低速测试。")

    def close_serial(self):
        self.emergency_stop()
        with self.io_lock:
            if self.stream:
                self.stream.close()
            self.stream = None
        self.connect_button.configure(text="连接")
        self.status.set("已断开")

    def _set_motion_buttons(self, enabled: bool):
        for button in self.motion_buttons:
            button.configure(state="normal" if enabled else "disabled")

    def start_motion(self, motion: str):
        if not self.stream or not self.stream.is_open:
            messagebox.showwarning("尚未连接", "请先连接下位机串口。")
            return
        if self.busy:
            return
        duration = self.duration.get()
        if duration <= 0:
            messagebox.showerror("时长错误", "单次时长必须大于 0。")
            return
        self.busy = True
        self.stop_event.clear()
        self._set_motion_buttons(False)
        self.status.set(f"正在{motion}… 可随时点击“紧急停车”。")
        packet = encode_packet(motion_state(self.config_data, motion, self.speed.get()))
        threading.Thread(target=self._run_motion, args=(packet, duration), daemon=True).start()

    def _run_motion(self, packet: bytes, duration: float):
        try:
            with self.io_lock:
                if self.stream and self.stream.is_open:
                    send_for_duration(self.stream, packet, duration, self.stop_event)
                    self.stream.write(encode_packet(motion_state(self.config_data, "stop", 0)))
                    self.stream.flush()
        except (OSError, serial.SerialException) as exc:
            self.after(0, lambda: messagebox.showerror("串口写入失败", str(exc)))
        finally:
            self.after(0, self._motion_finished)

    def _motion_finished(self):
        self.busy = False
        self._set_motion_buttons(True)
        if not self.stop_event.is_set():
            self.status.set("动作完成，履带已自动停车。")

    def emergency_stop(self):
        self.stop_event.set()
        # The worker owns io_lock while a normal motion is sent.  The event
        # stops its next 100 ms heartbeat; then its guaranteed stop packet runs.
        self.status.set("已请求紧急停车…")

    def close(self):
        self.close_serial()
        self.destroy()


if __name__ == "__main__":
    TrackDebugApp().mainloop()
