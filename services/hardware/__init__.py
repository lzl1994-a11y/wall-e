"""Concrete device adapters, connection management, and wire protocols.

English
-------
Hardware services are the only place that should understand USB selection,
serial probing/reconnection, ESP32 network configuration, PCA9685 channel
encoding, physical servo limits, or board-specific I2C behavior.  They accept
already-decided targets from motion or node boundaries and report device-level
results.  Business intent, dialog wording, and action selection do not belong
here.  A future CAN-servo implementation should be added behind this boundary,
not spread through motion, dialog, or ROS callback code.

中文
----
硬件服务是唯一应该理解 USB 设备选择、串口探测与重连、ESP32 配网、PCA9685 通道编码、
物理舵机限制和板级 I2C 差异的位置。它接收运动层或节点边界已经决定好的目标，并返回
设备级结果；用户意图、对话措辞和动作选择不属于本层。未来增加 CAN 总线舵机时，应在
这个边界后增加适配实现，而不是把 CAN 细节扩散到运动策略、对话流程或 ROS 回调中。
"""
