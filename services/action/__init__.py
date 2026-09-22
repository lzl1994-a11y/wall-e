"""Action contracts, arbitration, cancellation, and execution state.

动作层负责统一命令、仲裁、取消和执行状态；这里描述“要执行什么”，不直接管理
ROS 节点生命周期，也不访问舵机、串口等具体设备。
"""
