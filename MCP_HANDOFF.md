# Wali X3 MCP 改造交接文档

更新时间：2026-08-24
项目目录：`E:\walle\wall-ubuntu\wall-e`

## 1. 改造目标

在不影响机器人现有实时语音对话的前提下，为外部 Agent 增加标准 MCP 接入能力。

目标链路：

```text
内部语音：用户 → ASR/多模态 LLM → Function Calling → 安全校验 → ROS

外部 Agent：用户 → 外部 LLM/Agent → MCP Streamable HTTP
                                      ↓
                              Wali MCP Gateway
                                      ↓
                         参数校验 → /action_cmd → ROS
                                      ↓
                         /action_status → MCP 调用结果
```

关键原则：

- 不替换现有内部语音 `Function Calling → ROS` 链路。
- MCP 作为独立、可选的外部入口，默认关闭。
- 不暴露任意 ROS Topic、Service、Shell 或底层电机参数。
- 所有外部动作只能调用经过白名单和 Schema 限制的高层工具。
- MCP Server 不保存对话历史；多轮对话由外部 Agent 自己管理。
- 简单动作不需要第二次 LLM 请求，MCP 返回结构化执行结果即可。

## 2. 采用的技术方案

- MCP 实现：FastMCP 2.x。
- 网络协议：Streamable HTTP，新实现不使用旧 SSE transport。
- ROS 通信：MCP 网关作为独立 ROS 2 节点运行。
- 鉴权：局域网监听必须使用 Bearer Token，令牌只从环境变量读取。
- HTTP 模式：stateless HTTP。
- 工具输入：严格 JSON Schema 与运行时白名单双重校验。
- 工具结果：通过 `request_id` 和 `/action_status` 与 ROS 执行状态关联。

## 3. 当前已经完成的代码

### 3.1 MCP Server 与工具白名单

新增 `services/mcp_gateway.py`：

- 加载并校验 MCP 配置。
- 非回环地址监听时强制要求 `WALI_MCP_TOKEN`。
- Token 必须为 32～512 位 ASCII 且不能包含空白。
- 创建带鉴权的 FastMCP Server。
- 设置 `strict_input_validation=True` 与 `mask_error_details=True`。
- 为物理动作添加 MCP Tool Annotations。
- 暴露以下六个工具：
  - `move_chassis(direction, duration)`
  - `play_sequence(sequence_name)`
  - `express_emotion(emotion)`
  - `set_tracking_mode(mode)`
  - `set_vision_gate(enabled)`
  - `stop_all()`
- `move_chassis.duration` 在 MCP Schema 中限制为 1～3 秒。

未暴露 `inspect_camera`。当前摄像头观察流程与内部 LLM/预览生命周期绑定，还没有适合外部 MCP 的独立执行服务，不能返回虚假的成功结果。

### 3.2 MCP 到 ROS 的执行桥

新增 `nodes/wali_mcp_server.py`：

- 启动 FastMCP Streamable HTTP Server。
- 创建 ROS Publisher `/action_cmd`。
- 订阅 `/action_status`。
- 每次调用生成唯一 `request_id`。
- 等待动作所有者返回终态。
- 无 ROS 动作订阅者时返回 `ros_action_owner_unavailable`。
- 超时未收到终态时返回 `timeout`，不会误报成功。

### 3.3 动作请求与状态协议

修改 `services/action_command.py`：

- `build_action_cmd()` 新增可选 `request_id` 和 `source`。
- 新增 `new_action_request_id()`。
- 新增 `parse_action_request()`，保留关联元数据。
- 原 `parse_action_cmd()` 保持兼容，旧调用方无需修改。

新增 `services/action_status.py`：

- 状态 Topic：`/action_status`。
- 支持状态：
  - `accepted`
  - `running`
  - `completed`
  - `rejected`
  - `failed`
  - `interrupted`
- 提供状态编码和解析函数。

### 3.4 ROS 动作执行回执

修改 `nodes/sequence_ros_node.py`：

- 识别带 `request_id` 的动作请求。
- 为底盘、预设动作、表情和停止动作发布关联状态。
- 底盘到达持续时间后才返回 `completed`。
- 新命令中断旧 MCP 动作时返回 `interrupted`。
- 未知预设动作返回 `rejected`。
- 非本节点负责的跟踪和摄像头工具会直接忽略，不再错误地中断当前预设动作。
- `stop_all` 会停止 MCP 控制的底盘动作并中断当前预设序列。

修改 `nodes/wali_tracking_node.py`：

- 使用统一动作请求解析。
- `set_tracking_mode` 和 `set_vision_gate` 发布关联状态。
- 无效跟踪模式返回 `rejected`。

### 3.5 参数验证

修改 `services/action_intent_guard.py`：

- 新增 `validate_action_arguments()`，供已授权的 MCP 结构化调用使用。
- 外部 MCP 不信任客户端伪造的自然语言 `user_text`。
- 内部语音继续使用 `validate_action_call(user_text, ...)` 检查用户原话是否明确要求动作。
- 新增 `stop_all` 空参数验证。

### 3.6 启动与配置

修改 `launch_nodes.py`：

- 新增 `--mcp`，临时启用 MCP。
- 新增 `--no-mcp`，覆盖配置并禁用 MCP。
- MCP 节点在 `sequence_ros_node` 之后启动。

本机 `core/config.yaml` 已加入以下配置，但该文件被 `.gitignore` 忽略，不会出现在 Git diff 中：

```yaml
mcp:
  enabled: false
  host: 127.0.0.1
  port: 5555
  path: /mcp
  command_timeout_sec: 12.0
```

修改 `services/web_server.py`，增加 MCP 配置字段校验。

更新文档：

- `README.md`
- `ROS_NODES.md`

### 3.7 新增测试

- `tests/test_action_protocol.py`
- `tests/test_mcp_gateway.py`
- `tests/test_launch_nodes.py` 增加 MCP 启停测试。

## 4. 当前运行方式

### 4.1 仅本机测试

保持配置：

```yaml
mcp:
  enabled: false
  host: 127.0.0.1
  port: 5555
  path: /mcp
  command_timeout_sec: 12.0
```

运行：

```bash
python3 launch_nodes.py --mcp
```

本机 MCP URL：

```text
http://127.0.0.1:5555/mcp
```

### 4.2 局域网 Agent 接入

修改配置：

```yaml
mcp:
  enabled: true
  host: 0.0.0.0
  port: 5555
  path: /mcp
  command_timeout_sec: 12.0
```

生成并设置 Token：

```bash
export WALI_MCP_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 launch_nodes.py
```

客户端配置：

```text
URL: http://<机器人局域网IP>:5555/mcp
Authorization: Bearer <WALI_MCP_TOKEN>
Transport: Streamable HTTP
```

不要把 Token 写入 `core/config.yaml`、README、日志或 Git。

## 5. 已完成验证

### 5.1 自动化测试

以下相关测试共 60 项，全部通过：

```bash
python -m unittest \
  tests.test_action_protocol \
  tests.test_mcp_gateway \
  tests.test_mcp_tools \
  tests.test_action_intent_guard \
  tests.test_launch_nodes \
  tests.test_motion_arbiter \
  tests.test_sequence_motor_heartbeat \
  tests.test_wali_tracking_node -v
```

`python -m compileall -q services nodes launch_nodes.py tests` 通过。

`git diff --check` 通过，仅有 Git 的 LF/CRLF 提示。

### 5.2 全量测试情况

全量测试共 294 项。MCP 改造相关测试通过；有 5 项既有 `test_esp32_netcfg` 测试失败，因为当前 Windows 开发机没有连接配置中的 ESP32 USB 串口。这些失败单独复跑仍可复现，不经过本次 MCP 代码。

### 5.3 HTTP 冒烟测试

已使用假执行器启动真实 Streamable HTTP Server，并使用 Bearer Token 完成：

- MCP 初始化。
- `tools/list`。
- `move_chassis(direction="forward", duration=1)` 的 `tools/call`。
- 收到结构化 `completed` 结果。

该测试没有连接或驱动真实机器人硬件。

## 6. 尚未完成与已知限制

### 6.1 必须进行旭日 X3 真机联调

当前开发环境是 Windows，没有 `rclpy` 和真实 TogetherROS/硬件。必须在机器人上验证：

- FastMCP 2.x 在 ARM/X3 环境可以正常启动。
- `/action_cmd` 与 `/action_status` Topic 正常发现。
- `move_chassis` 1 秒后返回 `completed`。
- 新动作中断旧动作时旧调用返回 `interrupted`。
- 手柄控制仍然具有高于 autonomy/MCP 的优先级。
- 跟踪节点关闭时，跟踪工具应超时或失败，不能误报完成。
- MCP进程退出不会影响现有语音和 ROS控制节点。

### 6.2 当前鉴权的适用范围

目前使用环境变量静态 Bearer Token，适合受信局域网和 PoC。

禁止直接暴露公网。正式公网部署应增加：

- HTTPS。
- VPN 或反向代理。
- 正式 OAuth/JWT 验证和令牌轮换。
- IP/设备白名单。
- 调用频率限制和审计日志。

### 6.3 `stop_all` 的边界

当前 `stop_all` 停止的是 MCP/autonomy 控制的底盘动作与当前预设序列。它不会覆盖手柄，因为手柄必须保持最高优先级；它也不是硬件急停。

真正的物理急停必须继续由底层 MCU、运动仲裁器或实体开关实现，不能依赖 LLM 或 MCP。

### 6.4 摄像头工具尚未开放

`inspect_camera` 仍只在内部语音链路中使用。若要开放给外部 Agent，应先将“获取单帧、视觉分析、结果回传”拆成独立 ROS Service/Action，并定义图片大小、超时、并发和隐私策略。

### 6.5 动作完成的语义

- 底盘 `completed`：持续时间结束并发布停止指令。
- 预设动作 `completed`：时间轴已执行，舵机到达目标，且关联电机动作停止。
- 表情和跟踪切换 `completed`：相关 ROS 节点已接受并完成状态切换，不代表外部环境目标已经达到。

## 7. 下一会话建议任务

建议新会话按以下顺序继续：

1. 在旭日 X3 上安装当前依赖并运行相关单元测试。
2. 使用 `host: 127.0.0.1` 启动 MCP，先做本机 `tools/list`。
3. 在不接通电机或架空履带的条件下测试 `play_sequence`、`express_emotion` 和 `stop_all`。
4. 架空履带测试 `move_chassis(forward, 1)` 的状态时序。
5. 测试并发调用：第二个动作应中断第一个，旧调用返回 `interrupted`。
6. 测试手柄接管期间 MCP 底盘调用的实际效果与返回语义。
7. 确认稳定后再切换 `host: 0.0.0.0`，使用 Token 从局域网客户端连接。
8. 根据真机结果决定是否增加 `/robot_status` 和 ROS Action 接口。

## 8. 推荐真机验收用例

### 工具发现

- 无 Token 连接局域网端点应失败。
- 正确 Token 能列出且只能列出六个白名单工具。
- `move_chassis.duration=4` 应在 MCP Schema 层被拒绝。
- 未知工具应被拒绝。

### 运动安全

- 前进 1 秒后自动停车并返回 `completed`。
- 前进 3 秒期间调用 `stop_all`，前一个请求返回 `interrupted`。
- 前进期间使用手柄接管，手柄保持最高优先级。
- MCP Server断开或崩溃时，运动仲裁器超时停车。

### 普通对话兼容性

- MCP关闭时，原有 ASR→LLM→TTS 正常。
- MCP开启时，普通聊天请求次数和回复行为不变。
- 内部语音动作仍经过 `validate_action_call(user_text, ...)`。
- 外部 MCP调用不允许通过伪造 `user_text` 绕过参数限制。

## 9. Git 交接检查

本次 MCP 改造已纳入版本控制。新会话开始后先运行：

```bash
git status --short
git diff --check
```

本次改造涉及文件：

```text
README.md
ROS_NODES.md
launch_nodes.py
nodes/sequence_ros_node.py
nodes/wali_tracking_node.py
nodes/wali_mcp_server.py
services/action_command.py
services/action_intent_guard.py
services/action_status.py
services/mcp_gateway.py
services/web_server.py
tests/test_action_protocol.py
tests/test_launch_nodes.py
tests/test_mcp_gateway.py
MCP_HANDOFF.md
```

`core/config.yaml` 被 Git 忽略，但本机文件已加入默认关闭的 MCP 配置段。不要覆盖其中其他现有硬件和服务配置。

## 10. 给下一会话的简短指令

可直接把下面内容发给新会话：

> 请先完整阅读项目根目录 `MCP_HANDOFF.md`，然后检查当前 Git diff。不要重写现有语音链路，也不要暴露任意 ROS Topic。继续完成旭日 X3 真机 MCP 联调，优先验证鉴权、tools/list、move_chassis 1 秒回执、stop_all 中断和手柄最高优先级；发现问题后修复并运行相关测试。
