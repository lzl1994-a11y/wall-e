"""Camera-frame, preview, tracking-display, and visual-search services.

视觉层负责图像帧、预览、视觉检索和跟踪画面处理。它使用抽象的帧与管线协议，物理
摄像头的生命周期仍由节点边界管理，避免算法逻辑绑定具体采集设备。
"""
