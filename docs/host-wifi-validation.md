# 配网实机验证记录

日期：2026-10-19。设备：X3 Ubuntu 22.04.4，`wlan0` / `brcmfmac`，
NetworkManager 1.36.6；USB TFT ESP32-S3，MAC `e0:72:a1:d3:9e:30`。

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
主机 C++ `test_host_network_screen.cpp`、原 WTFT/NETCFG 协议测试和 PlatformIO 构建通过。
启动器套件另有一条既有失败：
`test_remote_gateway_uses_pyav_resampler_before_ndarray_conversion` 检查旧版
`converted.to_ndarray()` 文本；其余 37 项通过，本次没有改动 WebRTC 音频代码。

没有通过实体手机申请 DHCP 租约，也未直接观察屏幕或现场听音；
上述屏幕确认来自固件 ACK，语音确认来自播放节点日志。
只验证了当前真实 2.4 GHz 网络和不存在网络的恢复，未实测另一网段、酒店认证页或企业 Wi-Fi。
