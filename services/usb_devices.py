"""USB device discovery and stable role selection for WALI peripherals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "core" / "config.yaml"
USB_SYSFS = Path("/sys/bus/usb/devices")
USB_ROLES = ("camera", "screen_motion", "voice")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _class_members(class_name: str, pattern: re.Pattern[str]) -> list[Path]:
    root = Path("/sys/class") / class_name
    try:
        return [path for path in root.iterdir() if pattern.fullmatch(path.name)]
    except OSError:
        return []


def _belongs_to(member: Path, usb_device: Path) -> bool:
    try:
        resolved_member = member.resolve()
        resolved_usb = usb_device.resolve()
    except OSError:
        return False
    return resolved_member == resolved_usb or resolved_usb in resolved_member.parents


def _selector_for(device: dict[str, Any]) -> dict[str, str]:
    selector = {
        "vendor_id": device["vendor_id"],
        "product_id": device["product_id"],
    }
    if device.get("serial_number"):
        selector["serial_number"] = device["serial_number"]
    elif device.get("port_path"):
        # Devices without serial numbers are tied to a physical USB socket.
        selector["port_path"] = device["port_path"]
    return selector


def _device_id(selector: dict[str, str]) -> str:
    suffix = selector.get("serial_number") or selector.get("port_path") or "any"
    return f"{selector['vendor_id']}:{selector['product_id']}:{suffix}"


def _linux_usb_devices() -> list[dict[str, Any]]:
    if not USB_SYSFS.is_dir():
        return []

    tty_members = _class_members("tty", re.compile(r"tty(?:ACM|USB)\d+"))
    video_members = _class_members("video4linux", re.compile(r"video\d+"))
    sound_members = _class_members("sound", re.compile(r"card\d+"))
    devices: list[dict[str, Any]] = []

    for sys_path in sorted(USB_SYSFS.iterdir(), key=lambda item: item.name):
        if ":" in sys_path.name or not (sys_path / "idVendor").is_file():
            continue
        vendor_id = _read_text(sys_path / "idVendor").lower()
        product_id = _read_text(sys_path / "idProduct").lower()
        if vendor_id == "1d6b":
            continue

        serial_number = _read_text(sys_path / "serial")
        manufacturer = _read_text(sys_path / "manufacturer")
        product = _read_text(sys_path / "product")
        serial_ports = [f"/dev/{item.name}" for item in tty_members if _belongs_to(item, sys_path)]
        video_devices = [f"/dev/{item.name}" for item in video_members if _belongs_to(item, sys_path)]
        audio_cards = [int(item.name[4:]) for item in sound_members if _belongs_to(item, sys_path)]

        device: dict[str, Any] = {
            "vendor_id": vendor_id,
            "product_id": product_id,
            "serial_number": serial_number,
            "port_path": sys_path.name,
            "manufacturer": manufacturer,
            "product": product,
            "bus_number": _read_text(sys_path / "busnum"),
            "device_number": _read_text(sys_path / "devnum"),
            "interfaces": {
                "serial": sorted(serial_ports),
                "video": sorted(video_devices),
                "audio_cards": sorted(audio_cards),
            },
        }
        selector = _selector_for(device)
        device["id"] = _device_id(selector)
        device["selector"] = selector
        name = product or manufacturer or "USB device"
        details = [f"{vendor_id}:{product_id}"]
        if serial_number:
            details.append(f"SN {serial_number}")
        else:
            details.append(f"port {sys_path.name}")
        device["label"] = f"{name} ({', '.join(details)})"
        devices.append(device)
    return devices


def _serial_usb_devices() -> list[dict[str, Any]]:
    """Portable fallback used when Linux sysfs is unavailable."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for port in serial.tools.list_ports.comports():
        if port.vid is None or port.pid is None:
            continue
        vendor_id = f"{port.vid:04x}"
        product_id = f"{port.pid:04x}"
        serial_number = port.serial_number or ""
        port_path = port.location or ""
        key = (vendor_id, product_id, serial_number, port_path)
        if key not in grouped:
            device = {
                "vendor_id": vendor_id,
                "product_id": product_id,
                "serial_number": serial_number,
                "port_path": port_path,
                "manufacturer": port.manufacturer or "",
                "product": port.product or port.description or "",
                "bus_number": "",
                "device_number": "",
                "interfaces": {"serial": [], "video": [], "audio_cards": []},
            }
            selector = _selector_for(device)
            device["id"] = _device_id(selector)
            device["selector"] = selector
            device["label"] = f"{device['product'] or 'USB device'} ({vendor_id}:{product_id})"
            grouped[key] = device
        grouped[key]["interfaces"]["serial"].append(port.device)
    return list(grouped.values())


def list_usb_devices() -> list[dict[str, Any]]:
    devices = _linux_usb_devices()
    return devices if devices else _serial_usb_devices()


def load_usb_selector(
    role: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, str] | None:
    if role not in USB_ROLES:
        raise ValueError(f"Unknown USB role: {role}")
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    usb_devices = config.get("usb_devices")
    if not isinstance(usb_devices, dict):
        return None
    selector = usb_devices.get(role)
    return selector if isinstance(selector, dict) and selector else None


def selector_matches(selector: dict[str, Any], device: dict[str, Any]) -> bool:
    if str(selector.get("vendor_id", "")).lower() != device.get("vendor_id"):
        return False
    if str(selector.get("product_id", "")).lower() != device.get("product_id"):
        return False
    serial_number = str(selector.get("serial_number", "")).strip()
    if serial_number:
        return serial_number == device.get("serial_number")
    port_path = str(selector.get("port_path", "")).strip()
    return not port_path or port_path == device.get("port_path")


def find_selected_usb_device(
    role: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, Any] | None, bool]:
    selector = load_usb_selector(role, config_path)
    if selector is None:
        return None, False
    for device in list_usb_devices():
        if selector_matches(selector, device):
            return device, True
    return None, True


def serial_ports_for_role(
    role: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[list[str], bool]:
    device, configured = find_selected_usb_device(role, config_path)
    if not configured:
        return [], False
    if device is None:
        return [], True
    return list(device["interfaces"].get("serial", [])), True


def resolve_camera_device(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str | None:
    device, configured = find_selected_usb_device("camera", config_path)
    if configured:
        if device is None:
            return None
        videos = device["interfaces"].get("video", [])
        return videos[0] if videos else None

    camera_index = 0
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        camera_index = int(config.get("vision", {}).get("camera_index", 0))
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        pass
    default_path = Path(f"/dev/video{camera_index}")
    return str(default_path) if default_path.exists() else None


@dataclass(frozen=True)
class AudioDeviceResolution:
    configured: bool
    available: bool
    index: int | None
    identity: str


def resolve_audio_device(
    kind: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    sounddevice_module: Any = None,
) -> AudioDeviceResolution:
    if kind not in {"input", "output"}:
        raise ValueError("kind must be input or output")
    device, configured = find_selected_usb_device("voice", config_path)
    if not configured:
        return AudioDeviceResolution(False, True, None, "system-default")
    if device is None:
        return AudioDeviceResolution(True, False, None, "selected-usb-offline")

    if sounddevice_module is None:
        import sounddevice as sounddevice_module

    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    audio_cards = set(device["interfaces"].get("audio_cards", []))
    product_tokens = [
        token.lower()
        for token in (device.get("product", ""), device.get("manufacturer", ""))
        if len(token.strip()) >= 3
    ]
    candidates: list[tuple[int, int, str]] = []
    try:
        devices = sounddevice_module.query_devices()
    except Exception:
        return AudioDeviceResolution(True, False, None, device["id"])

    for index, audio_device in enumerate(devices):
        if int(audio_device.get(channels_key, 0)) <= 0:
            continue
        name = str(audio_device.get("name", ""))
        lowered = name.lower()
        score = 0
        for card in audio_cards:
            if re.search(rf"\bhw:{card}(?:,|\b)", lowered) or f"card={card}" in lowered:
                score = max(score, 100)
        if product_tokens and any(token in lowered for token in product_tokens):
            score = max(score, 60)
        if score:
            candidates.append((score, index, name))

    if not candidates:
        return AudioDeviceResolution(True, False, None, device["id"])
    _, index, name = max(candidates)
    return AudioDeviceResolution(True, True, index, f"{device['id']}:{index}:{name}")
