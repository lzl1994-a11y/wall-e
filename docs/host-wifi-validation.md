# 配网实机验证记录

日期：2026-10-19。设备：X3 Ubuntu 22.04.4，`wlan0` / `brcmfmac`，
NetworkManager 1.36.6；USB TFT ESP32-S3，MAC `e0:72:a1:d3:9e:30`。

## 文件变更清单

上位机修改/新增 22 个文件，无删除：

- 网络管理：`services/hardware/host_network.py`。
- 串口：`services/hardware/esp32_netcfg.py`、`esp32_netcfg_rpc.py`、`serial_bridge.py`，
  `nodes/serial_ros_node.py`。
- 监听与启动：`nodes/tft_tcp_service_node.py`、`launch_nodes.py`。
- 网页：`services/integrations/web_server.py`、`web/config/index.html`、`wifi.html`、`wifi.js`。
- 语音：`generate_wake_response.py`、`services/audio/esp32_network_prompt.py`、
  `assets/wifi_setup.wav`、`assets/wifi_connected.wav`。
- 测试：`tests/test_host_network.py`、`test_serial_netcfg_startup.py`、`test_system_audio.py`、
  `test_tft_tcp_service_node.py`、`test_web_server.py`。
- 文档：`docs/host-wifi.md`、`docs/host-wifi-validation.md`。

下位机修改/新增 8 个文件，无删除：

- `src/domain/HostNetworkScreen.h`。
- `src/ports/IDisplayPort.h`。
- `src/adapters/display/St7789DisplayPort.h`、`St7789DisplayPort.cpp`。
- `src/app/AppController.h`、`AppController.cpp`。
- `tools/test_host_network_screen.cpp`、`docs/host-wifi.md`。

两端任务开始前已有的未提交改动及图像资产均保留，没有纳入本次提交。

## 部署

- 上位机：`codex/behavior-tree-orchestration`，代码经 Git bundle fetch + fast-forward 更新到
  `/root/wall-e-bt`；源代码没有直接复制或在设备上修改。GitHub 同步推送。
- 下位机：源码提交 `cdfdea7`，`pio run -e esp32` 构建成功；固件 830080 字节。
- 固件 SHA-256：`6731d1b4989e533bd8cc5c34149b068cd7a57a97828db51a77461a0be4f94429`。
- 与现有分区表比较一致后，仅向 `0x10000` 写应用固件，esptool 校验哈希通过。
- 设备备份：`/root/wall-e-tft-build/host-wifi-cdfdea7/previous-app.bin` 和
  `previous-partitions.bin`；本次构建固件在同一目录。
- 安装热点 DHCP 所需的 `dnsmasq-base`；使用已有 `walle.service` 运行主程序。

## 实际检查

- 四个 `hostnet` 状态均收到固件 `HOSTSCREEN` ACK；NETCFG 查询仍可正常工作。
- `iw` 确认 wlan0 进入 AP 模式，分配 `10.42.0.1`。
- dnsmasq 启动并提供 `10.42.0.10–10.42.0.254` DHCP 地址池。
- `http://10.42.0.1:8080/wifi` 返回 HTTP 200。
- 提交不存在的 SSID 后自动恢复原热点，状态带有失败原因。
- 提交当前有效网络后经历 `connecting → connected → syncing → ready`。
- ROS 播放节点记录 `esp32_net_host_configuring` 和 `esp32_net_host_connected`。
- ESP32 重新建立 WTFT TCP 连接，图像节点收到 `HELLO device_id=WALL_E_TFT`。
- 主机及 ESP32 联网成功后恢复原 Wi-Fi；8080/9000 监听所有接口，主程序 active。
- 临时恢复定时器已停止；没有留下自动切网任务。

端到端记录：设备 `/tmp/walle-host-wifi-e2e2.log`，最终 `E2E_PASS`。
运行日志：`journalctl -u walle.service`。

## 自动验证及范围

Windows 上运行以下检查，124 项通过：

```bash
python -m pytest tests/test_host_network.py tests/test_system_audio.py tests/test_esp32_netcfg.py tests/test_serial_netcfg_startup.py tests/test_serial_bridge_hot_path.py tests/test_web_server.py -q --tb=short
node --check web/config/wifi.js
```

板上单独运行 `tests/test_tft_tcp_service_node.py`，10 项通过。
最终板上组合运行 host-network、system-audio、NETCFG、串口启动/路由、TFT 节点测试，
57 项通过；实际保存配置后确认私密 YAML 和新增 NetworkManager keyfile 权限均为 `0600`。
主机 C++ `test_host_network_screen.cpp`、原 WTFT/NETCFG 协议测试和 PlatformIO 构建通过。
启动器套件另有一条既有失败：
`test_remote_gateway_uses_pyav_resampler_before_ndarray_conversion` 检查旧版
`converted.to_ndarray()` 文本；其余 37 项通过，本次没有改动 WebRTC 音频代码。

没有通过实体手机申请 DHCP 租约，也未直接观察屏幕或现场听音；
上述屏幕确认来自固件 ACK，语音确认来自播放节点日志。
只验证了当前真实 2.4 GHz 网络和不存在网络的恢复，未实测另一网段、酒店认证页或企业 Wi-Fi。

## 首次配网问题修复与复测

用户清空网络后反馈“扫描无效、主机联网但 TFT 停在设置页”。实机确认
`esp32_network` 为 null；原流程联网后直接读取 `.get("wifi")`，在保存和同步前中断。
新增测试先复现该错误，再验证 null/缺省配置都能完成首次配网。
页面原来未继承设置页令牌，截图中的请求因空令牌被拒绝；现在热点接口/子网的访问者
打开配网页面时获得仅用于配网 API 的短期 HttpOnly/SameSite 会话。局域网管理权限不变。

本次上位机修改/新增文件：`services/hardware/host_network.py`、`esp32_netcfg.py`、
`services/integrations/web_server.py`、`web/config/wifi.html`、`wifi.js`、`wifi.css`、
`tests/test_host_network.py`、`test_web_server.py`、`test_wifi_frontend.py`、
`test_wifi_frontend.cjs`、`docs/host-wifi.md`，以及本验证记录。
下位机修改文件：`src/domain/HostNetworkScreen.h`、`src/app/AppController.h`、
`AppController.cpp`、`src/adapters/display/St7789DisplayPort.cpp`、
`tools/test_host_network_screen.cpp`、`docs/host-wifi.md`。

上位机修复提交 `0d0bf46`；下位机提交 `c494b01`。新固件 SHA-256：
`a25271afec8e29715be1ac38cfc7c70b4ae03510b8ad61b570a5d01411423c94`。
已备份当前应用并烧录 `0x10000`，哈希验证通过，恢复 `walle.service`。

本地执行以下命令，128 项通过；板上针对首次配网和会话权限边界的 13 项通过。

```bash
python -m pytest tests/test_host_network.py tests/test_wifi_frontend.py tests/test_system_audio.py tests/test_esp32_netcfg.py tests/test_serial_bridge_hot_path.py tests/test_serial_netcfg_startup.py tests/test_web_server.py -q --tb=short
node --test tests/test_wifi_frontend.cjs
pio run -e esp32
```

板上 `/tmp/walle-wifi-blank-e2e.log` 记录初始配置确实为 null，
热点会话读取状态成功、无会话访问被拒绝、热点会话访问 `/api/config` 被拒绝。
扫描得到 7 个网络并找到当前 SSID；经热点会话提交后最终状态 `ready`，
凭据保存、ESP32 同步和 TFT done ACK 均成功，最后 `E2E_PASS`。
该 HTTP 会话测试由本机绑定热点地址模拟热点客户端，未替代实体手机的现场确认。
TFT 现在显示同步中、同步失败及重试地址；成功提示保持三秒后返回电量页。
测试后恢复原 Wi-Fi，定时恢复任务已停止，未再次删除用户配置。
