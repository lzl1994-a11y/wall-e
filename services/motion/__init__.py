"""Hardware-neutral motion rules, target calculation, and safety state.

运动层负责摇杆映射、舵机目标、电机指令、仲裁与看门狗等纯规则。它输出规范化的
运动意图，不绑定某一种 PWM、CAN 或串口驱动，方便未来替换执行硬件。
"""
