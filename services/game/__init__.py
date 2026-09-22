"""Emulator sessions, game input, frame/audio adaptation, and policies.

游戏层集中管理模拟器会话、输入映射、音视频适配、菜单和退出策略。游戏规则通过
清晰接口连接显示与音频服务，不把 libretro 等实现细节带入 ROS 节点。
"""
