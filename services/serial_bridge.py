"""
下位机桥接线程：RDK X3 与 ESP32-S3 之间的二进制数据打包与解析。

"""

from time import sleep


def run_serial_bridge(bus) -> None:
    while True:
        # TODO: 打包 motion_commands，下发到底盘/UI 屏幕，并解析下位机回传数据。
        sleep(0.02)
