# Wali X3 Brain 🧠

这是一个基于 ROS2 (Humble) 和 地平线旭日 X3 派 (Sunrise X3 Pi) 打造的仿生机器人大脑中枢。本项目为“瓦力”机器人赋予了视觉追踪、多模态语音交互、手柄接管以及物理防碰撞的能力。

## 🌟 核心特性

- 🤖 **仿生视觉追踪 (AI BPU)**：深度整合地平线 TogetherROS，实现无缝的人体/人脸追踪。独创“虚假扭头”仿生视效及双舵机补偿仰俯算法。
- 🎙️ **多模态语音交互**：支持基于 Qwen-Omni 的端到端音频大模型交互，也支持传统的 STT(Paraformer) + LLM + TTS(EdgeTTS) 链路。
- 🎮 **零侵入物理接管**：通过高频发布中枢，使用游戏手柄可随时实现最高优先级的纯物理控制（摇杆差速底盘、线性扳机压感眼球），并自动屏蔽大模型的行动指令。
- 🛡️ **小脑安全守护**：底层 `sequence_ros_node` 提供 50Hz 动作插值平滑，并内置严格的物理干涉检测（例如转头时眼睛自动抬高防止碰撞）。

## ⚙️ 硬件与环境要求

1. **主板**：地平线 旭日 X3 派 (Sunrise X3 Pi)
2. **系统**：Ubuntu 22.04 LTS (预装地平线 TogetherROS Humble)
3. **摄像头**：支持普通 USB 摄像头 (UVC) 或官方 MIPI 排线摄像头。
4. **控制板**：基于 ESP32/Arduino，通过串口与旭日派通信，挂载 PCA9685 (舵机) 和 TB6612 (电机驱动)。
5. **手柄**：标准 USB 无线游戏手柄。

## 🚀 安装与配置

### 1. 旭日派 AI 模型安装
视觉节点强依赖于地平线 BPU 硬件加速模型，必须确保系统内已安装官方模型包：
```bash
sudo apt update
sudo apt install tros-dnn-node-example
```

### 2. 编译本项目
将代码克隆到你的 ROS2 工作空间（如 `~/your_workspace/src/`）下：
```bash
cd ~/your_workspace
colcon build --packages-select wali_x3_brain
source install/setup.bash
```

选择百度实时语音识别时，需要安装其 WebSocket 客户端依赖：

```bash
pip install websocket-client
```

## 🎮 启动指南

系统的启动分为**“视觉感知端”**和**“瓦力大脑端”**两个独立的部分。

### 配置网页

项目内置了一个零额外依赖的 `config.yaml` 配置网页。默认只监听本机：

```bash
python services/web_server.py
```

浏览器访问 `http://<旭日派IP>:8080`（本机调试也可用 `http://127.0.0.1:8080`）。默认监听所有网络接口，默认访问令牌为 `123456`。页面将 ASR、LLM、唤醒词、TTS 和系统提示词归在“对话模式”中，可选择 `ASR → LLM → TTS` 或多模态 LLM；ASR 可分别配置智谱、阿里云和百度智能云，切换厂商时只显示并提交该厂商字段。每张配置卡片独立保存。服务端只合并提交的模块，保存前仍会校验完整配置，再原子替换 `core/config.yaml`。已有 API Key 不会回传到网页，密钥输入框留空会保留原值。

“硬件”页面可以为三类物理 USB 分配角色：摄像头 USB、屏幕/运动 USB、语音 USB（麦克风、扬声器和声源定位）。点击“刷新设备”后选择当前设备并独立保存。配置使用 VID/PID 加序列号识别设备；没有序列号时使用物理 USB 端口路径，因此 `/dev/ttyACM*`、`/dev/video*` 或音频卡编号变化不会影响匹配。未配置某个角色时继续使用原有代码默认逻辑。运行中修改配置或拔插设备后，串口、DOA、音频和摄像头节点会自动重新发现并恢复，无需重启主程序。

网页保存后的配置结构示例：

```yaml
usb_devices:
  camera:
    vendor_id: "1234"
    product_id: "5678"
    serial_number: camera-001
  screen_motion:
    vendor_id: "303a"
    product_id: "1001"
    serial_number: screen-001
  voice:
    vendor_id: "abcd"
    product_id: "0001"
    port_path: 1-2
```

配置服务也会跟随日常使用的节点启动器自动启动，无需再单独运行一次：

```bash
python launch_nodes.py --real-stt
```

`launch_nodes.py` 会把同一个配置服务作为受管子进程启动，页面地址为 `http://<旭日派IP>:8080`，主程序退出时网页服务也会一起停止。传入 `--no-web` 可以禁用。独立调试和主程序运行不要同时占用同一个端口；需要并行运行时，可给独立服务指定其他端口，例如 `python services/web_server.py --host 127.0.0.1 --port 8765`。

默认局域网访问令牌为 `123456`。页面右上角可修改本次访问使用的令牌，并会在当前浏览器会话中记住。如果要修改服务端实际接受的令牌，在启动前覆盖环境变量：

```bash
export WALI_CONFIG_HOST=0.0.0.0
export WALI_CONFIG_TOKEN="换成你自己的ASCII令牌"
python services/web_server.py
```

让配置服务随主程序启动并开放到局域网（默认已经如此）：

```bash
export WALI_CONFIG_HOST=0.0.0.0
export WALI_CONFIG_TOKEN=123456
python launch_nodes.py --real-stt
```

打开 `http://<旭日派IP>:8080`，在页面右上角输入同一个令牌。大多数节点只在启动时读取配置，因此保存后需要在确保运动机构安全的前提下重启主脑。

### 第一步：启动视觉 AI 节点 (依赖地平线 BPU)
旭日派强大的地方在于自带视觉算法包。打开终端，输入以下命令：

**如果你使用 USB 摄像头 (UVC)：**
```bash
source /opt/tros/humble/setup.bash
export CAM_TYPE=usb
ros2 launch dnn_node_example dnn_node_example.launch.py
```
*(启动成功后，浏览器访问 `http://<旭日派IP>:8000` 即可查看带追踪框的实时画面)*

### 第二步：启动瓦力大脑中枢
打开一个新的终端：
```bash
source /opt/tros/humble/setup.bash
source ~/your_workspace/install/setup.bash

# 启动大脑并开启视觉跟随功能
ros2 launch wali_x3_brain launch_nodes.py --tracking
```

#### 启动参数选项 (`launch_nodes.py`)
- `--tracking`：开启视觉追踪节点（监听地平线 BPU 输出）。
- `--voice-chat`：使用大模型端到端语音交互（Qwen-Omni）。
- `--real-stt`：使用阿里云 Paraformer 语音转文字。
- `--keyboard-stt`：使用键盘输入文字模拟语音识别（调试用）。
- `--no-serial`：不启动串口硬件桥接节点（纯代码调试模式）。
- `--no-web`：不启动 `config.yaml` 配置网页。

## 🧠 核心架构说明

### 视觉双模式系统 (`wali_tracking_node.py`)
为了适配“摄像头纯靠脖子仰俯，左右平移全靠底盘转弯”的独特机械结构，设计了两种状态：
1. **模式 1 (跟随模式 - body_follow)**：纯底盘出击。根据身体大小前进后退，根据左右偏移差速旋转。同时脑袋 (`head_yaw`) 模拟转向产生“仿生看人”效果。
2. **模式 2 (注视模式 - face_follow)**：底盘锁定刹车。脖子双舵机 (`neck_top` & `neck_bottom`) 自动反向补偿实现仰俯追踪。丢失目标自动原地打转搜寻；仅识别到身体时自动上扬镜头寻找人脸。

### 手柄控制映射 (`joy_control_node.py`)
- **左摇杆**：控制履带底盘前进/后退/差速转向。
- **右摇杆**：控制头部 (`head_yaw` 和 脖子仰俯)。
- **L2/R2 扳机**：0~100% 线性控制左眼/右眼抬起高度，松开自动降下（还原呆萌感）。
- **十字键 / L1R1**：控制手臂和眉毛。自带 **3秒智能倒计时复位**，按下抬起，松开 3 秒后自动降回原位。
- **A/B/X/Y 键**：一键触发预设宏剧本（高兴、失落、打招呼等）。

## 📝 消息话题 (Topics)
- `/hobot_dnn_detection`：来自地平线节点的 AI 识别框数据。
- `/servo_cmd`：底层的舵机驱动指令 (JSON)。
- `/motor_cmd`：底层的双履带电机驱动指令 (JSON)。
- `/action_cmd`：小脑 API，接收大模型和手柄下发的组合动作、模式切换和 `manual_servo` 直驱指令。
