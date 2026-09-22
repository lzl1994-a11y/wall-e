"""Hardware-neutral motion policies, target calculation, and safety state.

English
-------
This package converts joystick, tracking, and autonomous intent into normalized
servo targets or left/right motor commands.  It owns dead zones, range limits,
source priority, watchdog decisions, sequence interpolation, and mechanical
coordination rules.  It deliberately does not know whether a target is carried
by PWM, CAN, I2C, or a serial MCU.  Concrete conversion and device I/O belong in
``services.hardware`` so replacing an actuator bus does not rewrite motion policy.

中文
----
本包把手柄、视觉跟踪和自主行为意图转换为规范化舵机目标或左右履带命令，统一管理死区、
范围限制、控制源优先级、看门狗停车决策、动作序列插值和机械联动约束。它刻意不知道目标
最终通过 PWM、CAN、I2C 还是串口下位机发送；数值编码和设备 I/O 必须留在
``services.hardware``，从而在更换执行总线时无需重写运动策略。
"""
