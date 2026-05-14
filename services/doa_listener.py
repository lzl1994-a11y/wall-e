"""
听觉监听线程：循环读取 ESP32_DOA 传来的 UART 角度数据，定位声源。
"""

from time import sleep


def run_doa_listener(bus) -> None:
    while True:
        # TODO: 读取 UART 角度数据，并写入 bus.doa_events。
        sleep(0.05)
