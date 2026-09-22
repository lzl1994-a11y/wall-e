"""Multi-step plans, behavior trees, and conditional workflow coordination.

English
-------
Orchestration turns one high-level goal into an ordered, cancellable execution
workflow.  It validates plans, coordinates behavior-tree leaves, normalizes
execution results, and decides whether a conditional or visual step may
continue.  It depends on :mod:`services.action` contracts and injected executor
callbacks.  It must never bypass action arbitration to write motors, servos, or
display devices; transport-specific ROS Action code is kept in a small adapter.

中文
----
编排层把一个高层目标拆成有顺序、可取消、可汇报状态的执行流程，负责计划校验、行为树
叶节点协调、执行结果归一化，以及条件任务和视觉步骤是否继续的决策。它依赖
``services.action`` 的动作契约，并通过注入的执行回调访问外部能力。编排代码不得绕过
动作仲裁直接写电机、舵机或显示设备；与 ROS Action 相关的传输细节只保留在小型适配器中。
"""
