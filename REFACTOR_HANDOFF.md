# Wali 架构重构交接

更新时间：2026-08-25
项目目录：`E:\walle\wall-ubuntu\wall-e`

## 1. 本轮结论

项目的核心价值应是 **机器人运行时与能力服务**，而不是自建一个与现成 Agent 重叠的 LLM 框架。

目标不是删除 LLM，而是把它替换为可交换的轻量 Agent：

- 内置 Agent：支持本机语音入口与默认模型。
- 外部 Agent：Codex、Hermes 或其他 MCP 客户端。
- 两者调用同一套机器人能力，最终落到同一批 ROS 服务提供者和安全机制。

## 2. 目标架构

```text
语音入口 A
  唤醒词 / VAD / ASR / 会话状态 / 自动播报
        ↓
LightAgent
  模型适配 / 历史 / 多轮任务循环 / 工具调用 / 步数与超时限制
        ↓
Skill Registry + Capability Facade
   ├─ 本地直接调用（内置 Agent）
   └─ MCP Server（外部 Agent）
        ↓
服务层
  视觉跟随、相机、动作、底盘、屏幕、TTS、媒体、状态、安全
        ↓
ROS 节点层
  Topic / Service / Action 适配、硬件驱动与消息收发
```

**安全与仲裁**独立于 Agent：手柄优先级、限速、看门狗、急停、运动仲裁必须始终在 ROS/硬件侧生效。

## 3. 各层职责

### 语音入口 A

负责本地人机对话生命周期，不负责规划或直接控制硬件：

- 本地唤醒、VAD、ASR。
- 创建 `turn_id`，把文本（及必要的音频/图片）交给 `AgentPort`。
- 接收 Agent 的最终文字回复，自动转给 TTS 和播放服务。
- 在播放期间处理回声抑制、暂停/恢复监听和 UI 状态。

普通对话的播报不应让 Agent 每次调用两个工具（“生成语音”再“播放音频”）；这是 A 的默认工作流。

### LightAgent

负责：模型调用、历史、多轮工具循环和任务收敛；不做 PID、摄像头连续帧处理或电机时序。

应支持受限的循环：

```text
观察（拍照/状态）→ LLM 决策 → 调用高层能力 → 等结果 → 再观察
```

例如“去远处风扇旁拍一张照”需要多次推理与拍照，但每次移动必须是短时、高层、可停止的动作。设置最大步数、每步超时、总超时；失败或超限时调用 `stop_all` 并向用户说明。

建议采用 **PydanticAI** 作为 Agent/工具循环框架；可选 **LiteLLM** 只用于多供应商模型适配。避免嵌入 nanobot 一类完整产品，也不要让 `CodeAgent` 获得可执行任意代码的机器人控制权限。

需要定义统一模型接口，例如：

```text
ModelProvider.stream(messages, tools, images, audio) -> LLMEvents
```

并维护每个模型的能力档案：文本、视觉、音频、工具调用、流式输出。没有可靠工具调用能力的模型不能执行物理动作。

### Skill Registry / Capability Facade

这里的 Skill 是“面向 Agent 的能力合同”，不是把每个 ROS Topic 原样暴露。

- **Skill**：说明何时可用、参数、结果、停止方式、权限和安全限制。
- **Capability Facade（薄服务调度器）**：把统一能力调用转到对应服务提供者或 ROS Action。
- **MCP Tool**：Skill 给外部 Agent 的可调用入口。

内置 Agent 应直接调用 Facade；外部 Agent 通过 MCP Server 调用同一个 Facade。不要让内置 Agent 绕 HTTP 回环调用 MCP，也不要维护两套业务逻辑。

### 服务层与 ROS 节点层

- 服务提供者实现业务能力和状态机。
- 节点发布者/接收者只负责 ROS Topic、Service、Action、定时器和硬件适配。
- 高实时闭环始终留在 ROS 内部。

当前项目是部分分离：`sequence_ros_node.py`、`wali_tracking_node.py`、`llm_ros_node.py`、`voice_chat_ros_node.py` 仍包含较多业务流程；这是后续抽取的重点。

## 4. 对外 Skill 建议

第一版只暴露有清晰业务语义的高层能力：

```text
视觉跟随： start_visual_follow / get_visual_follow_status / stop_visual_follow
视觉采集： capture_image / save_photo / inspect_image（可选）
动作表演： perform_gesture / express_emotion / stop_action
底盘控制： move_chassis_short / stop_motion
显示：     display_text / display_expression
媒体：     speak / play_media / stop_media / get_media_status
机器人：   get_robot_status / stop_all
```

不要直接公开舵机角度、底层话题、任意 shell、PID 参数或持续裸速度指令。

“视觉跟随”应该是机器人内部 Behavior：Agent 只发启动、查询、停止；相机帧、BPU、PID、目标丢失恢复和运动控制都由 ROS 服务/节点完成。长期行为后续建议使用 ROS Action，而不是把“模式已切换”误当作“跟随任务已完成”。

## 5. 当前代码事实

- 实际外部 MCP 网关：`services/mcp_gateway.py` + `nodes/wali_mcp_server.py`。
- 目前仅公开 6 个工具：`move_chassis`、`play_sequence`、`express_emotion`、`set_tracking_mode`、`set_vision_gate`、`stop_all`。
- `inspect_camera` 仅在内部 LLM 工具集，尚未对外公开。
- 相机拍照由 `services/camera_frame.py` / `nodes/camera_capture_node.py` 管理；保存目录来自 `tft_preview.photo_directory`，当前默认 `~/.wali/photos`，文件名形如 `wali_TIMESTAMP_UUID.jpg`。
- 当前 `raise_hand` 序列会在约 2 秒后放下手；要保持举手应调用 `arms_up`。
- `nodes/wali_mcp_server.py` 已通过 `/action_cmd` 和 `/action_status` 等待终态回执。这是未来 Capability Facade 的可复用基础，但不应在其中加入具体跟随/PID等业务逻辑。
- 只有单独启动 MCP Server 时，没有动作所有者会返回 `ros_action_owner_unavailable`；真机应运行完整的 `python3 launch_nodes.py --mcp`。
- 机器人真机已验证：`http://192.168.0.6:5555/mcp` 可连通、工具发现正常、`play_sequence` 可返回完成状态。历史测试 Token 已在对话中暴露，继续使用前必须重新生成并轮换。

详情见已有 [MCP_HANDOFF.md](MCP_HANDOFF.md)。

## 6. 重构顺序（禁止大爆炸）

1. 定义统一的 Skill Registry：名称、JSON Schema、权限、风险级别、结果和停止语义。
2. 为每个关键能力定义服务接口与状态模型，优先相机、视觉跟随、动作、底盘、播放、机器人状态。
3. 从 ROS 节点抽取业务服务：节点保留 ROS 适配，服务承载能力状态机。
4. 实现本地 `CapabilityFacade`；MCP Gateway 改为只调用它。
5. 引入 LightAgent 与 `ModelProvider`；先替换 `llm_ros_node.py` 中的供应商适配和工具循环，保持 A 的语音行为不变。
6. 给 MCP 增加 `capture_image`、`get_robot_status`、行为 `start/status/stop`；完成权限、图片返回策略和真机验证后再公开。
7. 最后逐步退役旧的内部直接 Function Calling 路径，确保始终只有一份能力实现。

## 7. 新会话起始指令

> 请先阅读 `REFACTOR_HANDOFF.md` 和 `MCP_HANDOFF.md`，检查 `git status --short`。目标是把 Wali 重构为“语音入口 A → 可替换 LightAgent → 统一 Skill/Capability Facade → 服务层 → ROS 节点层”。不要大爆炸重写；先设计并落地 Skill Registry、服务接口和 Capability Facade。内置 Agent 直调 Facade，外部 Agent 走 MCP，但最终使用同一服务实现。保留 ROS 侧安全仲裁和高频控制，不暴露原始 ROS Topic 或底层硬件接口。
