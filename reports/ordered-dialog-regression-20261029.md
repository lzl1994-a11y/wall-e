# 组合动作与视觉任务回归记录

测试日期：2026-10-29（Asia/Shanghai）。

代码改为由模型提出完整的 run_ordered_task 步骤，先整体校验再执行。物理动作复用原有 BT；观察、图片保存和原生视觉搜索保持顺序。

## 真实模型规划测试

使用运行端当前模型配置 Pro/moonshotai/Kimi-K2.6；仅请求和校验计划，未连接动作执行端、未启动机器人、未发送硬件命令。11/11 通过。

| 对话 | 结果 |
|---|---|
| 你低头，然后看看有什么。 | head_down → inspect_camera；通过 |
| 先低头，然后再看看前面有什么。 | head_down → inspect_camera；通过 |
| 先低头，挥手，然后看看前面有什么，并拍照保存。 | head_down → wave_hello → inspect_camera，save_photo=true；通过 |
| 先点头，再挥手，最后看看桌面上有什么。 | basic_nod → wave_hello → inspect_camera；通过 |
| 先转身，然后找一下红色盒子。 | move_chassis → search_environment；通过 |
| 先向左转一秒，然后找一下红色盒子，找到后点头。 | move_chassis → search_environment，找到后点头留在搜索树条件分支；通过 |
| 把“低头然后看看”翻译成英文。 | 普通回答，无执行工具；通过 |
| 请复述：先低头然后看看前面有什么。 | 普通回答，无执行工具；通过 |
| 先抬起右手，然后看看前面有什么。 | right_hand_up → inspect_camera；通过 |
| 先看看前面有什么，然后低头再看一眼地面。 | inspect_camera → head_down → inspect_camera；通过 |
| 你会低头然后看看吗？ | 普通回答，无执行工具；通过 |

## 执行层自动化测试

完整 tests 目录回归：1286 项通过，8 项按环境跳过，1 项已知旧断言单独排除。依赖按 requirements.txt 中的 av==17.1.0、aiortc==1.15.0 安装至临时测试目录，没有修改全局 Python 环境。

对话、动作授权、BT、视觉搜索、TTS/历史及工具注册相关回归：394 项通过。动作回执、摄像头和原生搜索使用模拟接口；图片保存用例真实写入并核对临时 JPEG 文件。

另外覆盖：残缺计划、反序步骤、参数非法、错误工具分流、动作仅 accepted 未 completed、失败/超时/取消、模型遗漏低头或保存要求、摄像头失败保留已完成动作历史、人物低头等视觉内容不转成机器人动作。

## 验证边界

- 转向按现有时长控制；本轮不验证精确 180° 或实际舵机到位。
- 直接音频模式不暴露尚未接入的组合任务工具；本轮修复针对运行端当前 ASR→LLM 流程。
- 本地原模型接口返回 Unsupported model，真实规划验证使用运行端配置快照完成；没有修改两端模型配置。
- 在仓库根目录直接运行 pytest 会收集 tools/diagnostics 中的手工硬件脚本；正式回归以 tests 目录为范围。
- 已知旧断言 test_remote_gateway_uses_pyav_resampler_before_ndarray_conversion 与本次改动无关，完整套件运行时单独排除。

可复跑：`python scripts/check_ordered_dialog_planning.py --live --config <有效项目配置路径>`。脚本不加载 ROS、摄像头或动作传输，不会执行生成的计划。
