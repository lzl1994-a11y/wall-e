"""Emulator sessions, controller input, media adaptation, and game policies.

English
-------
Game services wrap libretro lifecycle, controller state, hotkeys, menu choices,
frame/audio conversion, TFT streaming, commentary scheduling, and safe exit
barriers.  Nodes only translate ROS messages and timers into these APIs.  The
package may use display/audio protocols but must not let emulator-specific data
structures leak into general dialog, motion, or hardware services.

中文
----
游戏层封装 libretro 生命周期、手柄状态、组合热键、菜单选择、音视频转换、TFT 推流、
画面解说调度和安全退出屏障。节点只负责把 ROS 消息和定时器转换为这些 API 调用。游戏层
可以使用显示和音频协议，但不能让模拟器专用数据结构渗入通用对话、运动或硬件服务。
"""
