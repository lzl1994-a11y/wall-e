# 上位机热点配网

运行前需要 NetworkManager、`iw`、`dnsmasq-base`，并允许运行配置服务的用户管理
NetworkManager 连接和写入 `/etc/NetworkManager/system-connections`。X3 当前以 root
运行。配网热点只在 2.4 GHz 工作；第一版支持普通 WPA/WPA2 PSK 与开放网络。

1. 在设置页硬件卡片点击“上位机 Wi-Fi 配置”，或打开 `http://<上位机IP>:8080/wifi`。
2. 输入与设置页相同的访问令牌。点击“重新进入热点配网”。这会切断现有无线连接。
3. 胸前屏幕显示热点名称、随机密码及 `10.42.0.1:8080/wifi`（端口跟随网页服务）。
   手机连接该热点，选择保留无互联网连接，然后打开屏幕地址。
4. 读取状态、扫描网络或手动输入 SSID，提交密码。请求立即返回，后台稍后切换网络。
5. 成功后播放“wifi配置成功，请稍后”，胸前屏幕展示新 IP；手机切至目标网络。
   上位机保存本次凭据与新 IP，再通过原有 NETCFG SET/APPLY 同步 ESP32。
6. Wi-Fi 连接失败会恢复原配网热点；ESP32 同步失败保留上位机连接，可点击单独重试。

主程序中的 Web 服务会监视无线连接，持续约 45 秒未连接后进入配网热点；
有局域网连接但没有互联网时不切换。普通单网卡采用 AP/STA 切换，手机短暂断开是正常的。
本次配网会持久保存一个救援热点配置，但禁止它开机抢占已知 Wi-Fi；
目标网络由 NetworkManager 保存并启用自动连接。
进入配网时会将原来绑定某个 LAN IP 的 TFT 图像服务改为 `0.0.0.0`，并通过
`host_network_prepare/host_network_ready` 等待运行中服务完成重绑；该设置会持久保存，
保证跨网段和重启后仍可接收 ESP32 的 TCP HELLO。

Wi-Fi 密码仅写入 root 可读的 NetworkManager keyfile 和原有私密配置。
网页状态、日志和 ROS 状态消息均不包含密码。热点密码只通过串口展示到设备屏幕。
现有访问令牌保护同样适用于所有新增 API：

| API | 用途 |
| --- | --- |
| GET `/api/host-network/status` | 安全状态快照 |
| GET `/api/host-network/scan` | 附近 SSID、信号和加密方式 |
| POST `/api/host-network/setup` `{}` | 异步进入配网 |
| POST `/api/host-network/connect` `{"ssid":"...","password":"..."}` | 异步连接并同步 |
| POST `/api/host-network/retry-sync` `{}` | 重试当前进程最近一次 ESP32 同步 |

## 串口扩展

新增命令，不改变 NETCFG v2 SET/APPLY/QUERY：

```text
hostnet:<seq>|1|<state>|<IPv4>|<web_port>|<AP_SSID>|<AP_password>\r\n
NETCFG:RESULT:<seq>|HOSTSCREEN|<result>|0
```

`state` 为 `setup`、`connecting`、`connected` 或 `done`；后两项允许空字符串。
命令使用有界 ASCII 字段，禁止 `|` 和控制字符；序列号为 uint32。
结果 `0` 表示主屏已处理命令，`3` 表示参数错误。
配网页面优先于聊天、视频和网络状态页脚，`done` 才返回电量界面。
旧固件会超时，必须先更新 TFT 固件；主程序启动前预生成两条 48 kHz 单声道提示音：

```bash
python generate_wake_response.py wifi_setup.wav wifi_connected.wav
```

验证：`python -m pytest tests/test_host_network.py tests/test_system_audio.py
tests/test_esp32_netcfg.py tests/test_serial_netcfg_startup.py tests/test_web_server.py -q`。
真机还需检查热点广播、DHCP、输入错误密码后的恢复、目标网络连接、ESP32 HELLO、
屏幕不被视频覆盖和实际扬声器播报。仅软件测试不能代替手机与屏幕的现场检查。
