"""Camera frames, previews, runtime validation, and visual-search workflows.

English
-------
Vision services define camera lease/frame protocols, convert and validate image
artifacts, manage preview workers, prepare tracking display frames, and execute
visual-search decisions.  The physical camera process and ROS subscriptions
remain node-owned; consumers request frames through the shared protocol instead
of opening the camera independently.  This guarantees a single capture owner
and keeps visual reasoning testable with injected frames.

中文
----
视觉层定义摄像头租约与帧协议，负责图像产物转换和校验、预览工作进程、跟踪显示帧以及
视觉搜索决策。物理摄像头进程和 ROS 订阅仍由节点持有；所有消费者必须通过共享协议请求
画面，不能各自打开摄像头。这样既保证唯一采集所有者，也能通过注入测试帧独立验证视觉
推理和工作流。
"""
