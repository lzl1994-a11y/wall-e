"""Multi-step plan, behavior-tree, and conditional workflow orchestration.

编排层负责把多个动作组织成计划、行为树或条件任务。它依赖动作层的稳定契约，
但不应绕过动作接口直接操作硬件，从而让编排策略可以独立测试和替换。
"""
