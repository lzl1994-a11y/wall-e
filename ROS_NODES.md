# ROS Nodes

这个文件记录当前工程里已经定义的 ROS 2 节点、默认启动方式、订阅/发布的话题，以及每个节点的作用。

信息来源：`nodes/*.py` 里的 `super().__init__(...)` 节点名，以及 `launch_nodes.py` 的启动列表。

## 默认启动链路

默认执行：

```bash
python launch_nodes.py
```

启动链路是：

```text
keyboard_stt_test_node -> voice_text -> walle_llm_brain -> screen_dialog -> walle_serial_node
```

如果使用真实语音识别：

```bash
python launch_nodes.py --real-stt
```

链路变成：

```text
walle_ear_node -> voice_text -> walle_llm_brain -> screen_dialog -> walle_serial_node
```

## ROS 节点清单

| 脚本 | ROS 节点名 | 默认启动 | 订阅话题 | 发布话题 | 作用 |
| --- | --- | --- | --- | --- | --- |
| `nodes/keyboard_stt_node.py` | `keyboard_stt_test_node` | 是，默认测试输入 | 无 | `voice_text` | 键盘输入测试节点。你在终端输入文字后，它把文字发布到 `voice_text`，模拟 STT 输出。 |
| `nodes/stt_ros_node.py` | `walle_ear_node` | 否，使用 `--real-stt` 时启动 | 无 | `voice_text` | 真实语音识别节点。调用 `services/stt_service.py`，识别到一句话后发布到 `voice_text`。 |
| `nodes/llm_ros_node.py` | `walle_llm_brain` | 是 | `voice_text` | `corrected_text`, `tts_text`, `full_ai_text`, `action_cmd`, `screen_dialog` | 大模型大脑节点。接收用户文本，调用 LLM 做纠错、回复、工具调用，并把结果分发给 TTS、屏幕和动作系统。 |
| `nodes/serial_ros_node.py` | `walle_serial_node` | 是，除非加 `--no-serial` | `screen_dialog` | 无 | 串口/屏幕输出节点。接收完整对话包，把用户文本、AI 回复和动作命令写给下位机或屏幕。 |
| `nodes/yolo_node.py` | `yolo_brain_node` | 否 | 无 | `/wall_e/vision` | 视觉演示节点。定时发布模拟视觉识别结果，目前不在 `launch_nodes.py` 默认链路里。 |

## 关键话题说明

| 话题 | 发布者 | 订阅者 | 作用 |
| --- | --- | --- | --- |
| `voice_text` | `keyboard_stt_test_node` 或 `walle_ear_node` | `walle_llm_brain` | 用户输入文本。调试“我说了什么/键盘输入了什么”时看这个。 |
| `corrected_text` | `walle_llm_brain` | 当前默认无人订阅 | LLM 纠正后的用户文本，比如把 `nihao` 纠正成 `你好`。 |
| `tts_text` | `walle_llm_brain` | 当前默认无人订阅 | 给 TTS 用的流式分句文本。适合边生成边播报，但不一定是完整回复。 |
| `full_ai_text` | `walle_llm_brain` | 当前默认无人订阅 | LLM 完整回复文本，等整轮生成结束后发布。 |
| `action_cmd` | `walle_llm_brain` | 当前默认无人订阅 | 单独的工具/动作命令通道，保留给动作执行节点使用。 |
| `screen_dialog` | `walle_llm_brain` | `walle_serial_node` | 一整轮完整对话包，包含 `turn_id`、`corrected_text`、`ai_text`、`actions`。目前屏幕串口节点主要看这个。 |
| `/wall_e/vision` | `yolo_brain_node` | 当前默认无人订阅 | 视觉识别结果演示话题。 |

### 视觉跟踪链路（--tracking）

使能跟踪:

```bash
python launch_nodes.py --tracking
```

此时在默认链路基础上附加:

```text
wali_tracking_node  -> /servo_cmd -> servo_ros_node  -> ServoControl (舵机PCA9685)
                    -> /motor_cmd -> motor_ros_node  -> ServoControl (电机TB6612)
        ^
        ├─ /hobot_mono2d_body_detection  (RDK BPU 感知)
        ├─ /action_cmd                    (LLM 模式切换)
        └─ /doa_angle  <- doa_ros_node <-> DOA串口
```

## 视觉跟踪节点清单

| 脚本 | ROS 节点名 | 启动条件 | 订阅话题 | 发布话题 | 作用 |
| --- | --- | --- | --- | --- | --- |
| `nodes/wali_tracking_node.py` | `wali_tracking_node` | `--tracking` | `/hobot_mono2d_body_detection`, `/action_cmd`, `/doa_angle` | `/servo_cmd`, `/motor_cmd` | 视觉跟踪中枢。接收 BPU 感知结果，运行 BODY_FOLLOW / FACE_FOLLOW 状态机，发布舵机和电机控制指令。 |
| `nodes/servo_ros_node.py` | `servo_ros_node` | `--tracking` | `/servo_cmd` | 无 | 舵机桥接节点，将 ROS 指令转发给 ServoControl.set_angle() 驱动 PCA9685 舵机。 |
| `nodes/motor_ros_node.py` | `motor_ros_node` | `--tracking` | `/motor_cmd` | 无 | 电机桥接节点，将 ROS 指令转发给 ServoControl.set_motor() 驱动 TB6612FNG 电机。 |
| `nodes/doa_ros_node.py` | `doa_ros_node` | `--tracking` (除非 `--no-doa`) | 无(串口直读) | `/doa_angle` | DOA 声源定位桥接节点，对接 D-DOA TDOA 模块串口，发布声源角度。 |

### 视觉跟踪话题

| 话题 | 发布者 | 订阅者 | 作用 |
| --- | --- | --- | --- |
| `/servo_cmd` | `wali_tracking_node` | `servo_ros_node` | JSON: `{"name":"head_yaw","angle":110}` |
| `/motor_cmd` | `wali_tracking_node` | `motor_ros_node` | JSON: `{"left":{"action":1,"throttle":30},"right":{...}}` |
| `/doa_angle` | `doa_ros_node` | `wali_tracking_node` | `std_msgs/Int32`，声源角度（°） |
| `/hobot_mono2d_body_detection` | RDK X3 `mono2d_body_detection` | `wali_tracking_node` | `ai_msgs/PerceptionTargets`，BPU 检测结果（body/face/head/hand 框 + track_id） |

### 跟随模式切换

LLM 解析用户语音指令后，通过 `/action_cmd` 下发:

```json
{"turn_id":"...","name":"set_tracking_mode","arguments":{"mode":"body_follow"}}
{"turn_id":"...","name":"set_tracking_mode","arguments":{"mode":"face_follow"}}
{"turn_id":"...","name":"set_vision_gate","arguments":{"enabled":true}}    // 默认 body_follow
{"turn_id":"...","name":"set_vision_gate","arguments":{"enabled":false}}   // 关闭跟踪
```

## 辅助服务文件

这些文件不是 ROS 节点，但被节点调用：

| 文件 | 作用 |
| --- | --- |
| `services/llm_service.py` | 封装 OpenAI/Kimi 兼容接口，提供流式大模型回复和工具调用结果。 |
| `services/mcp_service.py` | 注册可给 LLM 调用的工具，目前包括 `express_emotion`、`perform_action`、`move_chassis`。 |
| `services/stt_service.py` | 底层语音识别服务，被 `walle_ear_node` 调用。 |
| `services/serial_bridge.py` | 底层串口扫描和发送服务，被 `walle_serial_node` 调用。 |
| `services/serial_broker.py` | 串口设备挂载/管理相关逻辑。 |

## 常用调试命令

查看当前运行的节点：

```bash
ros2 node list
```

查看所有话题：

```bash
ros2 topic list
```

看用户输入：

```bash
ros2 topic echo /voice_text
```

看一整轮 LLM 输出：

```bash
ros2 topic echo /screen_dialog
```

看 TTS 分句输出：

```bash
ros2 topic echo /tts_text
```
