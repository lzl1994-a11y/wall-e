"""NetworkManager provisioning and recovery; serial access stays in SerialNode."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid

from services.hardware.esp32_netcfg import NetworkConfigError, validate_network_payload


AP_ADDRESS = "10.42.0.1"
AP_PROFILE = "WALL-E-Setup"


def validate_wifi(ssid, password):
    # Keep UTF-8 byte limits authoritative in NETCFG.
    settings = validate_network_payload({
        "wifi": [{"ssid": ssid, "password": password}] + [{"ssid": "", "password": ""}] * 2,
        "host": AP_ADDRESS, "port": 9000,
    })
    if any(ord(char) < 32 or ord(char) == 127 for value in (ssid, password) for char in value):
        raise NetworkConfigError("Wi-Fi 名称和密码不能含控制字符")
    if password and not (8 <= len(password.encode("utf-8")) <= 63 or
                         re.fullmatch(r"[0-9a-fA-F]{64}", password)):
        raise NetworkConfigError("Wi-Fi 密码需为 8–63 字节或 64 位十六进制密钥")
    return settings.wifi[0]


def split_nmcli(line):
    """Decode terse NetworkManager fields, including escaped colons/backslashes."""
    fields, value, escaped = [], [], False
    for char in line:
        if escaped:
            value.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(value))
            value = []
        else:
            value.append(char)
    fields.append("".join(value))
    return fields


class NetworkManagerBackend:
    """Small nmcli adapter. Credentials go only into root-owned keyfiles."""

    def __init__(self, *, runner=subprocess.run,
                 profile_dir="/etc/NetworkManager/system-connections"):
        self.runner = runner
        self.profile_dir = Path(profile_dir)
        self.device = None

    def run(self, *args, timeout=12):
        try:
            result = self.runner(["nmcli", "--colors", "no", *args],
                                 capture_output=True, text=True, timeout=timeout,
                                 check=False, env={**os.environ, "LC_ALL": "C"})
        except (OSError, subprocess.SubprocessError) as exc:
            raise NetworkConfigError("NetworkManager 调用失败或超时") from exc
        if result.returncode:
            # Never forward command output: it can contain credentials.
            raise NetworkConfigError("网络操作失败，请检查密码、信号和网卡状态")
        return result.stdout.strip()

    def status(self):
        rows = self.run("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status")
        for row in rows.splitlines():
            fields = split_nmcli(row)
            if len(fields) != 4 or fields[1] != "wifi":
                continue
            if self.device and fields[0] != self.device:
                continue
            self.device = fields[0]
            connected = fields[2] == "connected"
            address = ""
            if connected:
                values = self.run("-g", "IP4.ADDRESS", "device", "show", self.device)
                for value in values.splitlines():
                    try:
                        address = str(ipaddress.IPv4Interface(value).ip)
                        break
                    except ValueError:
                        continue
            return {"device": self.device, "connected": connected,
                    "hotspot": fields[3] == AP_PROFILE, "ip": address}
        raise NetworkConfigError("没有找到可用的无线网卡")

    def scan(self):
        self.status()
        output = self.run("-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list",
                          "ifname", self.device, "--rescan", "auto")
        networks = {}
        for row in output.splitlines():
            fields = split_nmcli(row)
            if len(fields) != 3 or not fields[0] or not fields[1].isdigit():
                continue
            entry = {"ssid": fields[0], "signal": int(fields[1]), "security": fields[2]}
            if entry["signal"] > networks.get(entry["ssid"], {}).get("signal", -1):
                networks[entry["ssid"]] = entry
        return sorted(networks.values(), key=lambda item: -item["signal"])

    @staticmethod
    def _escape(value):
        return value.replace("\\", "\\\\").replace(" ", "\\s")

    def _profile(self, ssid, password, *, hotspot=False):
        self.status()
        identifier = str(uuid.uuid4())
        name = AP_PROFILE if hotspot else f"WALL-E-{identifier}"
        # SSID as a byte array avoids keyfile syntax ambiguity for user input.
        ssid_bytes = "".join(f"{value};" for value in ssid.encode("utf-8"))
        body = (f"[connection]\nid={name}\nuuid={identifier}\ntype=wifi\n"
                f"interface-name={self.device}\nautoconnect=false\n\n"
                f"[wifi]\nmode={'ap' if hotspot else 'infrastructure'}\nssid={ssid_bytes}\n")
        if hotspot:
            body += "band=bg\n"
        if password:
            body += ("\n[wifi-security]\nkey-mgmt=wpa-psk\npsk-flags=0\n"
                     f"psk={self._escape(password)}\n")
        body += (f"\n[ipv4]\nmethod={'shared' if hotspot else 'auto'}\n"
                 + (f"address1={AP_ADDRESS}/24\n" if hotspot else "")
                 + "\n[ipv6]\nmethod=disabled\n")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        path = self.profile_dir / f"walle-{identifier}.nmconnection"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                target.write(body)
            self.run("connection", "load", str(path))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return identifier

    def activate(self, identifier):
        self.run("--wait", "35", "connection", "up", "uuid", identifier,
                 "ifname", self.device, timeout=40)

    def connect(self, ssid, password):
        identifier = self._profile(ssid, password)
        try:
            self.activate(identifier)
            status = self.status()
            if not status["connected"] or status["hotspot"] or not status["ip"]:
                raise NetworkConfigError("Wi-Fi 未获得 IPv4 地址")
            self.run("connection", "modify", "uuid", identifier, "connection.autoconnect", "yes")
            return status
        except Exception:
            self.run("connection", "delete", "uuid", identifier)
            raise

    def hotspot(self):
        status = self.status()
        capable = self.run("-g", "WIFI-PROPERTIES.AP", "device", "show", self.device)
        if capable != "yes":
            raise NetworkConfigError("当前无线网卡不支持 AP 热点模式")
        # Keep the rescue profile across process/device restarts.
        profiles = self.run("-t", "-f", "UUID,NAME", "connection", "show")
        identifier = next((fields[0] for row in profiles.splitlines()
                           if len(fields := split_nmcli(row)) == 2 and fields[1] == AP_PROFILE), None)
        if identifier:
            ssid = self.run("-g", "802-11-wireless.ssid", "connection", "show", "uuid", identifier)
            password = self.run("--show-secrets", "-g", "802-11-wireless-security.psk",
                                "connection", "show", "uuid", identifier)
        else:
            ssid, password = "WALL-E-" + secrets.token_hex(2).upper(), secrets.token_hex(6)
            identifier = self._profile(ssid, password, hotspot=True)
        if not status["hotspot"]:
            self.activate(identifier)
        return {"ssid": ssid, "password": password, "ip": AP_ADDRESS}


class HostNetworkManager:
    """Own one asynchronous provisioning job and restore the AP on failures."""

    def __init__(self, store, rpc_factory, port, *, backend=None, switch_delay=2):
        self.store, self.rpc_factory, self.port = store, rpc_factory, port
        self.backend = backend or NetworkManagerBackend()
        self.switch_delay = switch_delay
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._worker = None
        self._monitor = None
        self._state = {"state": "idle", "detail": "", "ip": "", "esp32": "unknown"}
        self._ap = None
        self._last_payload = None

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _update(self, **values):
        with self._lock:
            self._state.update(values)

    def _screen(self, state, ip, *, audible=False):
        ap = self._ap or {}
        self.rpc_factory().host_screen({"state": state, "ip": ip, "port": self.port,
                                       "ssid": ap.get("ssid", ""),
                                       "password": ap.get("password", ""), "audible": audible})

    def _prepare_listener(self):
        # A listener pinned to the old LAN address cannot receive ESP32 HELLO
        # after a subnet change. The serial RPC asks its owner to rebind live.
        config = self.store.load()
        if config.get("tft_preview", {}).get("bind_address", "0.0.0.0") != "0.0.0.0":
            self.store.save_patch({"tft_preview": {"bind_address": "0.0.0.0"}})

    def _launch(self, task, *args):
        with self._lock:
            if self._closed.is_set():
                raise NetworkConfigError("配网服务已停止")
            if self._worker and self._worker.is_alive():
                raise NetworkConfigError("已有配网操作正在执行，请稍候")
            self._state.update(state="starting", detail="")
            self._worker = threading.Thread(target=task, args=args, daemon=True, name="host-network")
            self._worker.start()
        return self.snapshot()

    def enter_setup(self):
        return self._launch(self._setup)

    def connect(self, payload):
        if not isinstance(payload, dict):
            raise NetworkConfigError("配网请求必须是对象")
        credential = validate_wifi(payload.get("ssid"), payload.get("password"))
        return self._launch(self._connect, credential.ssid, credential.password)

    def retry_sync(self):
        if self._last_payload is None:
            raise NetworkConfigError("没有待同步的网络，请先完成 Wi-Fi 配置")
        return self._launch(self._sync)

    def _setup(self, detail=""):
        try:
            self._prepare_listener()
            self._ap = self.backend.hotspot()
            self._screen("setup", AP_ADDRESS, audible=True)
            self._update(state="setup", ip=AP_ADDRESS, detail=detail, esp32="pending")
        except NetworkConfigError as exc:
            self._update(state="failed", detail=str(exc))
        except Exception:
            self._update(state="failed", detail="热点或屏幕配网提示启动失败，请检查网卡、DHCP 依赖及串口服务")

    def _connect(self, ssid, password):
        # Let the HTTP accepted response reach the phone before switching wlan0.
        if self._closed.wait(self.switch_delay):
            return
        self._update(state="connecting", detail="正在连接目标 Wi-Fi，手机连接可能断开")
        try:
            self._prepare_listener()
            self._screen("connecting", AP_ADDRESS)
        except Exception:
            self._update(state="failed", detail="配网屏幕或图像服务尚未就绪，未切换 Wi-Fi，请稍后重试")
            return
        try:
            status = self.backend.connect(ssid, password)
        except Exception:
            self._setup("目标 Wi-Fi 连接失败，已恢复热点，请检查密码并重试")
            return
        self._update(state="connected", ip=status["ip"], esp32="pending", detail="Wi-Fi 已连接，正在同步 ESP32")
        try:
            config = self.store.load()
            previous = config.get("esp32_network", {}).get("wifi", [])
            others = [item for item in previous if item.get("ssid") and item["ssid"] != ssid]
            wifi = ([{"ssid": ssid, "password": password}] + others +
                    [{"ssid": "", "password": ""}] * 2)[:3]
            payload = {"wifi": wifi, "host": status["ip"],
                       "port": config.get("tft_preview", {}).get("port", 9000)}
            validate_network_payload(payload)
            self._last_payload = payload
            # Persist before ESP32 APPLY so a transient ESP failure can be retried after reboot.
            self.store.save_patch({"esp32_network": payload})
            self._screen("connected", status["ip"], audible=True)
        except Exception:
            self._update(state="partial", detail="Wi-Fi 已连接，但启动配置保存或屏幕提示失败；请重新提交配置")
            return
        self._sync()

    def _sync(self):
        try:
            self._update(state="syncing", esp32="pending")
            self.rpc_factory().save_and_apply(self._last_payload)
            self._screen("done", self.snapshot()["ip"])
            self._update(state="ready", esp32="connected", detail="上位机和 ESP32 网络配置完成")
        except Exception:
            self._update(state="partial", esp32="failed", detail="上位机已联网，ESP32 同步失败，可单独重试")

    def start(self):
        if os.name != "posix" or not shutil.which("nmcli"):
            return
        self._monitor = threading.Thread(target=self._watch, daemon=True, name="host-network-watch")
        self._monitor.start()

    def _watch(self):
        disconnected_since = None
        while not self._closed.wait(5):
            if self._worker and self._worker.is_alive():
                continue
            try:
                status = self.backend.status()
                if status["hotspot"]:
                    if self.snapshot()["state"] == "idle":
                        self.enter_setup()
                    disconnected_since = None
                elif status["connected"]:
                    disconnected_since = None
                    self._update(ip=status["ip"])
                else:
                    disconnected_since = disconnected_since or time.monotonic()
                    if time.monotonic() - disconnected_since >= 45:
                        self.enter_setup()
                        disconnected_since = None
            except NetworkConfigError:
                self._update(detail="无法读取无线网络状态，请检查 NetworkManager")

    def close(self):
        self._closed.set()
        # Finish an in-flight switch/recovery before closing the shared RPC transport.
        if self._worker:
            self._worker.join(timeout=100)
        if self._monitor:
            self._monitor.join(timeout=2)
