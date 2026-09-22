"""Concrete device adapters and low-level hardware protocol state.

硬件层封装串口、USB、PCA9685、ESP32 配网和舵机控制等具体实现。上层只通过明确
接口传入目标状态，设备协议、通道编码和连接细节均收敛在本目录。
"""
