# ROS Nodes

这个文件记录当前工程里已经定义的 ROS 2 节点、默认启动方式、订阅/发布的话题，以及每个节点的作用。

信息来源：`nodes/*.py` 里的 `super().__init__(...)` 节点名，以及 `launch_nodes.py` 的启动列表。

## 启动链路

不带语音模式参数时，启动器使用 `core/config.yaml` 中的 `pipeline.mode`。
如需在没有麦克风的环境中调试完整的 LLM/TTS 链路，执行：

```bash
python launch_nodes.py --keyboard-stt
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
| `nodes/keyboard_stt_node.py` | `keyboard_stt_test_node` | `pipeline.mode=keyboard` 或 `--keyboard-stt` | 无 | `voice_text` | 键盘输入测试节点。你在终端输入文字后，它把文字发布到 `voice_text`，模拟 STT 输出。 |
| `nodes/stt_ros_node.py` | `walle_ear_node` | `pipeline.mode=asr_llm` 或 `--real-stt` | 无 | `voice_text` | 真实语音识别节点。调用 `services/stt_service.py`，识别到一句话后发布到 `voice_text`。 |
| `nodes/llm_ros_node.py` | `walle_llm_brain` | 是 | `voice_text` | `corrected_text`, `tts_text`, `full_ai_text`, `action_cmd`, `screen_dialog` | 大模型大脑节点。接收用户文本，调用 LLM 做纠错、回复、工具调用，并把结果分发给 TTS、屏幕和动作系统。 |
| `nodes/music_player_node.py` | `music_player_node` | 是 | `/action_cmd`, `llm_busy`, `/game_mode_state` | `/music_audio`, `/music_spectrum`, `/music_state`, `/action_status` | 用 FFmpeg 解码本地音乐并发布 PCM 与频谱数据；不直接占用声卡或 TFT。 |
| `nodes/tft_tcp_service_node.py` | `tft_tcp_service_node` | 是 | `/tft_preview_request`, `/vision_pipeline_cmd`, `/game_mode_state`, `/game_frame`, `/music_state`, `/music_spectrum` | `/tft_preview_result`, `tft_preview_ready`, `/game_mode_request` | 胸前 TFT 的唯一 TCP 服务所有者；统一仲裁拍照、跟踪、游戏和音乐频谱画面。 |
| `nodes/serial_ros_node.py` | `walle_serial_node` | 是，除非加 `--no-serial` | `screen_dialog` | 无 | 串口/屏幕输出节点。接收完整对话包，把用户文本、AI 回复和动作命令写给下位机或屏幕。 |

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

### 摄像头热备与按需看图/拍照链路

`camera_capture_node` 始终启动，是 `/dev/video*` 的唯一生命周期管理者：

```text
LLM / 多模态 -> /tft_preview_request -> tft_tcp_service_node
                                             └-> /camera_capture_cmd -> camera_capture_node
                                                                          └-> 唯一 hobot_usb_cam
                                                                                    └-> /camera_frame

/camera_frame -> tft_tcp_service_node -> TFT TCP + /tft_preview_result
                                      ├-> 末帧视觉 LLM
                                      └-> 末帧本地照片
              -> Config Web preview
              -> 热备 BPU 检测进程（仅有租约时收到帧）
```

`camera_capture_node` 启动时即拉起唯一的 `hobot_usb_cam` 并持续检查首帧、断流和进程退出；
异常时自动重启。预览消费者只订阅 `/camera_frame`，不直接打开摄像头。客户端使用带超时的
租约来控制帧转发：`wali_tracking_node` 在跟随/注视期间持有独立租约，拍照和预览只增加、
释放自己的租约。全部租约释放或过期后停止向 `/camera_frame` 转发，但物理摄像头保持热备；
只有 `camera_capture_node` 退出时才关闭 `hobot_usb_cam`。

### 视觉跟踪链路（--tracking）

使能跟踪:

```bash
python launch_nodes.py --tracking
```

此时在默认链路基础上附加:

```text
wali_tracking_node  -> /servo_targets/tracking -> sequence_ros_node -> /servo_cmd -> 当前硬件后端
                    -> /motor_cmd/tracking ┐
sequence_ros_node   -> /motor_cmd/autonomy ├-> motion_arbiter_node -> /motor_cmd
joy_control_node    -> /motor_cmd/joystick ┘                         ├─ serial_mcu: hardware_bridge_node -> serial_ros_node -> ESP32
                                                                    └─ ubuntu_i2c: i2c_hardware_node -> 板载 I2C -> PCA9685
                    -> /vision_pipeline_cmd -> hobot_vision_node -> 热备 RDK BPU 检测（订阅 `/camera_frame`）
        ^
        ├─ /hobot_mono2d_body_detection  (RDK BPU 感知)
        ├─ /action_cmd                    (LLM 模式切换)
        └─ /doa_angle  <- doa_ros_node <-> DOA串口
```

## 视觉跟踪节点清单

| 脚本 | ROS 节点名 | 启动条件 | 订阅话题 | 发布话题 | 作用 |
| --- | --- | --- | --- | --- | --- |
| `nodes/camera_capture_node.py` | `camera_capture_node` | 始终 | `/camera_capture_cmd`, `/image` | `/camera_frame`, `/camera_capture_status` | 唯一物理摄像头所有者：保持 `hobot_usb_cam` 热备并自愈，按租约将 `/image` 适配为预览 JPEG。 |
| `nodes/wali_tracking_node.py` | `wali_tracking_node` | `--tracking` | `/hobot_mono2d_body_detection`, `/action_cmd`, `/doa_angle` | `/servo_targets/tracking`, `/motor_cmd/tracking`, `/vision_pipeline_cmd`, `/camera_capture_cmd` | 视觉跟踪中枢。跟随/注视时持有摄像头租约，并控制检测管线。 |
| `nodes/hobot_vision_node.py` | `hobot_vision_control` | `--tracking` | `/vision_pipeline_cmd`, `/camera_frame` | `/hobot_mono2d_body_detection` | 热备 RDK 编解码、补边和 `mono2d_body_detection`；跟踪命令只切换结果使用状态，不反复加载模型。 |
| `nodes/motion_arbiter_node.py` | `motion_arbiter_node` | 运动控制启用时 | `/motor_cmd/joystick`, `/motor_cmd/tracking`, `/motor_cmd/autonomy` | `/motor_cmd` | 唯一电机命令仲裁器，执行手柄 > 跟踪 > 自主动作的优先级，并在上游命令超时后停车。 |
| `nodes/hardware_bridge_node.py` | `hardware_bridge_node` | `hardware.backend=serial_mcu` | `/servo_cmd`, `/motor_cmd` | `/pca9685_raw` | 把舵机与电机状态合并后交给串口下位机；300ms 收不到仲裁心跳时强制写入停车状态。 |
| `nodes/i2c_hardware_node.py` | `i2c_hardware_node` | `hardware.backend=ubuntu_i2c` | `/servo_cmd`, `/motor_cmd` | 无 | 单实例持有板载 I²C，直接驱动 PCA9685；300ms 收不到仲裁心跳时直接停车。 |
| `nodes/doa_ros_node.py` | `doa_ros_node` | `--tracking` (除非 `--no-doa`) | 无(串口直读) | `/doa_angle` | DOA 声源定位桥接节点，对接 D-DOA TDOA 模块串口，发布声源角度。 |

### 视觉跟踪话题

| 话题 | 发布者 | 订阅者 | 作用 |
| --- | --- | --- | --- |
| `/camera_capture_cmd` | `CameraFrameProvider`, Web preview worker | `camera_capture_node` | JSON 租约命令：`acquire`、`renew`、`release`。 |
| `/image` | `hobot_usb_cam`（由 `camera_capture_node` 启动） | `camera_capture_node` | 摄像头唯一原始 JPEG 图像源，类型为 `sensor_msgs/msg/CompressedImage`。 |
| `/camera_frame` | `camera_capture_node` | TFT、Web preview、RDK 解码器 | 从 `/image` 按租约转发的 `sensor_msgs/msg/CompressedImage`；热备检测进程空闲时不收帧。 |
| `/camera_capture_status` | `camera_capture_node` | Web preview worker | 摄像头启动、复用、错误和当前客户端数量。 |
| `/servo_targets/tracking` | `wali_tracking_node` | `sequence_ros_node` | 深度为 1 的最新头颈目标；只更新插值目标，不打断高层动作。 |
| `/servo_cmd` | `sequence_ros_node` | 当前硬件后端 | JSON: `{"name":"head_yaw","pwm":5000}`，也兼容 `angle` |
| `/motor_cmd/joystick` | `joy_control_node` | `motion_arbiter_node` | 最高优先级手柄电机心跳。 |
| `/motor_cmd/tracking` | `wali_tracking_node` | `motion_arbiter_node` | 视觉跟踪电机心跳。 |
| `/motor_cmd/autonomy` | `sequence_ros_node` | `motion_arbiter_node` | LLM 与预设动作产生的电机心跳。 |
| `/motor_cmd` | `motion_arbiter_node` | 当前硬件后端 | 仲裁后的唯一电机输出；JSON: `{"left":{"action":1,"throttle":30},"right":{...}}`。 |
| `/doa_angle` | `doa_ros_node` | `wali_tracking_node` | `std_msgs/Int32`，声源角度（°） |
| `/hobot_mono2d_body_detection` | RDK X3 `mono2d_body_detection` | `wali_tracking_node` | `ai_msgs/PerceptionTargets`，BPU 检测结果（body/face/head/hand 框 + track_id） |
| `/vision_pipeline_cmd` | `wali_tracking_node` | `hobot_vision_control`, `tft_tcp_service_node` | `std_msgs/String`：`start`/`stop` 标记跟踪结果使用状态，并控制 TFT 跟踪预览；检测进程保持热备。 |
| `/tft_preview_request` | LLM 或多模态语音节点 | `tft_tcp_service_node` | 带 `request_id` 的一次性摄像头预览请求。 |
| `/tft_preview_result` | `tft_tcp_service_node` | 请求方 | 返回预览状态及未经 TFT 旋转的末帧，供视觉模型或照片保存使用。 |
| `/music_audio` | `music_player_node` | `audio_playback_node` | 48 kHz 单声道 int16 PCM；空消息表示音乐流结束。 |
| `/music_spectrum` | `music_player_node` | `tft_tcp_service_node` | 20 个归一化频段值，以 5 Hz 更新；上位机据此渲染 240×240 频谱帧。 |
| `/music_state` | `music_player_node` | `tft_tcp_service_node` | JSON 播放状态：`loading`、`playing`、`stopped` 或 `error`。 |

### 跟随模式切换

LLM 解析用户语音指令后，通过 `/action_cmd` 下发:

```json
{"turn_id":"...","name":"set_tracking_mode","arguments":{"mode":"body_follow"}}
{"turn_id":"...","name":"set_tracking_mode","arguments":{"mode":"face_follow"}}
{"turn_id":"...","name":"set_vision_gate","arguments":{"enabled":true}}    // 默认 body_follow
{"turn_id":"...","name":"set_vision_gate","arguments":{"enabled":false}}   // 关闭跟踪
```

进入跟随或注视模式时先申请摄像头；`/camera_capture_status` 确认收到有效帧后才允许使用 BPU 检测结果。物理摄像头与 BPU 检测进程均已热备，请求只打开 `/camera_frame` 帧转发。运行中连续 3 秒无有效帧时，摄像头管理节点发布错误并自动重启摄像头，检测进程保持存活并等待画面恢复。目标丢失 1 秒后开始慢速搜索，5 秒后停止搜索并原地等待；连续 60 秒未识别到目标则切回 `idle`，停止电机并关闭帧转发。多张人脸同时出现时，注视模式选择面积最大的人脸。

跟踪期间胸前 TFT 使用与游戏模式相同的持久流传输。整个跟踪会话只发送一次无限时长的
`STREAM_START`；内部摄像头租约分段续期不会再产生周期性的 `STREAM_END`，只有跟踪暂停、
退出或连接失败时才结束 TFT 流，从而防止 ESP32 在持续跟踪中误判空闲并休眠。

## 辅助服务文件

这些文件不是 ROS 节点，但被节点调用：

| 文件 | 作用 |
| --- | --- |
| `services/llm_service.py` | 封装 OpenAI/Kimi 兼容接口，提供流式大模型回复和工具调用结果。 |
| `services/camera_frame.py` | 请求摄像头租约，支持单帧或限时帧流，完成后立即释放；不会直接打开摄像头。 |
| `services/camera_capture_protocol.py` | 定义唯一摄像头源话题、租约 JSON、JPEG 转换和 `hobot_usb_cam` 启动命令。 |
| `services/tft_preview_server.py` | 后台监听 ESP32 TCP 连接，处理 WTFT 协议、心跳、240×240 JPEG 预览和断线重连。 |
| `services/tft_preview_client.py` | 对话节点使用的同步 ROS 客户端；向独立 TFT 节点请求预览并按 `request_id` 等待结果。 |
| `services/tft_preview_protocol.py` | 定义 TFT 预览请求/结果话题及与 ROS 无关的 JSON 编解码。 |
| `services/music_player.py` | 解析本地曲目、调用 FFmpeg 输出 PCM，并从同一音频块计算频谱。 |
| `services/music_spectrum.py` | 把频段值渲染为 TFT 原始帧；TCP 传输仍由 TFT 节点负责。 |
| `services/music_protocol.py` | 定义音乐 PCM、频谱和状态话题，避免播放与显示模块相互依赖。 |
| `services/mcp_service.py` | 以 FastMCP 2.x `get_tools()` 枚举 OpenAI function-calling 工具；枚举失败会明确报错，不会静默退化为无工具对话。 |
| `nodes/wali_mcp_server.py` | 可选的外部 Agent 网关；提供带鉴权的 Streamable HTTP MCP，将白名单工具转换为带回执的 `/action_cmd`。默认关闭。 |
| `services/mcp_gateway.py` | 外部 MCP 的配置、安全启动检查和工具白名单；不暴露任意 ROS Topic、Service 或 Shell。 |
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
