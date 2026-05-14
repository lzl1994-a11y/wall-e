"""
启动层：负责初始化配置、创建消息总线，并拉起所有进程/线程。
"""

from multiprocessing import Process
from threading import Thread

try:
    import yaml
except ImportError:
    yaml = None

from arbiter import Arbiter
from core.message_bus import create_message_bus
from services.doa_listener import run_doa_listener
from services.llm_service import run_llm_service
from services.mcp_service import run_mcp_service
from services.servo_control import run_servo_control
from services.serial_bridge import run_serial_bridge
from services.tts_service import run_tts_service
from services.vision_service import run_vision_service
from services.web_server import run_web_server


CONFIG_PATH = "core/config.yaml"


def load_config(path):
    if yaml is None:
        return {"config_path": path}

    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def main() -> None:
    config = load_config(CONFIG_PATH)
    bus = create_message_bus(config)

    vision_process = Process(
        target=run_vision_service,
        args=(bus,),
        name="vision-service",
        daemon=True,
    )

    threads = [
        Thread(target=run_llm_service, args=(bus,), name="llm-service", daemon=True),
        Thread(target=run_mcp_service, args=(bus,), name="mcp-service", daemon=True),
        Thread(target=run_tts_service, args=(bus,), name="tts-service", daemon=True),
        Thread(target=run_doa_listener, args=(bus,), name="doa-listener", daemon=True),
        Thread(target=run_servo_control, args=(bus,), name="servo-control", daemon=True),
        Thread(target=run_serial_bridge, args=(bus,), name="serial-bridge", daemon=True),
        Thread(target=run_web_server, args=(bus,), name="web-server", daemon=True),
    ]

    vision_process.start()
    for thread in threads:
        thread.start()

    Arbiter(bus).run()


if __name__ == "__main__":
    main()
