# ROS Node Architecture Audit

Date: 2026-09-18  
Scope: read-only review of every `nodes/*.py` entry point and the default
`launch_nodes.py` process graph.  This is an inventory, not authorization to
merge nodes or move code.

## Method and terminology

`P` and `S` below mean ROS publisher and subscription.  Service dependencies
are the project `services/` modules that contain the node's meaningful policy,
protocol, or device adapter; standard-library and ROS imports are omitted.
“Exclusive resource” means a resource that must have one lifecycle owner. A
topic publisher alone is not treated as exclusive.

Default launch means `launch_nodes.py` with its configured pipeline. Optional
means a CLI/configuration gate; utility means the script is present but is not
added by that launcher.  The two behavior-tree launcher scripts replace
themselves with installed C++ executables and are included for completeness.

## Inventory

| Entry point / ROS name | Launch | Interfaces (P / S) | Lifecycle and exclusive resource | Key service dependencies and remaining responsibility |
| --- | --- | --- | --- | --- |
| `action_coordinator_node.py` / `action_coordinator_node` | Default | P `/action_cmd`, `/action_cancel`, `/action_status`; S `/action_request`, `/action_status` | Timer expires action leases; owns high-level action arbitration and the single terminal-status decision for a preemption. | `action_arbitration`, action command/cancel/status protocols. Node is a thin ROS adapter around arbiter decisions. |
| `ai_msg_scaler_node.py` / `ai_msg_scaler` | Utility, not launcher-managed | P `/hobot_mono2d_body_detection` (parameterized); S `/image`, `/hobot_mono2d_body_detection_raw` | No device ownership; maintains latest image size/stamp. | No project service. Coordinate transform and clipping are pure logic, but this path is currently superseded by the padded detector pipeline. Do not extract or revive it without a real deployment need. |
| `audio_playback_node.py` / `audio_playback_node` | Default | P `llm_busy`, wake/system done; S `audio_output`, `/music_audio`, wake/system audio, ESP32 network status | Starts/stops one playback backend; **sole sound-card/output-stream owner** and mixer lifecycle. | `mixing_playback_service`, audio/wake/system/music protocols, network prompt selector. Keep separate from TTS generation. |
| `camera_capture_node.py` / `camera_capture_node` | Default | P `/camera_frame`, `/camera_capture_status`; S `/camera_capture_cmd`, `/image` | Starts, health-checks, restarts, and stops the single `hobot_usb_cam`; leases only gate frame relay. **Sole physical camera/V4L2 owner.** | `camera_capture_protocol`, `usb_devices`. Process and lease boundary correctly remain in Node. |
| `dialog_motion_node.py` / `dialog_motion_node` | Default with serial cluster | P dialog expression target; S `llm_busy`, `tts_text`, dialog expression | Timer emits speech/expression poses; no hardware access, but owns the dialog-motion presentation lifecycle. | expression and TTS protocols, `servo_motion_config`. `DialogPoseSampler` contains deterministic pose sampling and is the one future pure-service extraction candidate if this behavior expands. |
| `doa_ros_node.py` / `doa_ros_node` | Optional tracking; `--no-doa` disables | P `/doa_angle` | Reconnect timer opens/closes the D-DOA serial listener. **Sole DOA serial-device owner.** | `serial_broker`, `doa_listener`, `usb_devices`. Hardware bridge; keep isolated. |
| `game_mode_node.py` / `game_mode_node` | Default | P game state/frame, `audio_output`; S game mode request, `llm_busy` | Session thread owns emulator/game session and stream flushing. **Exclusive game/emulator session.** | `fc_game_session`, `game_mode`, game protocol. Controller owns state transitions; Node supplies ROS/audio effects. |
| `hardware_bridge_node.py` / `hardware_bridge_node` | Default only for `serial_mcu` hardware backend | P `/pca9685_raw`; S `/servo_cmd`, `/motor_cmd` | Flush timer combines state and enforces watchdog. Owns the downstream serial-MCU command boundary, not the serial port itself. | motor normalization/inversion and `motor_watchdog`. Hardware effect adapter; no merge with serial screen/NETCFG owner. |
| `hobot_vision_node.py` / `hobot_vision_control` | Optional tracking | S `/vision_pipeline_cmd`, `/image_nv12`, `/image_padded_nv12`; child processes publish detection | Starts/reaps codec, padder, and BPU detector group; command only enables result accounting. **Exclusive BPU detector process group.** | vision runtime/pipeline and camera protocols. Keep separate from camera owner because failure and restart boundaries differ. |
| `i2c_hardware_node.py` / `i2c_hardware_node` | Default only for `ubuntu_i2c` hardware backend | S `/servo_cmd`, `/motor_cmd` | Watchdog owns direct servo/motor output. **Sole I2C/PCA9685 owner** in this backend. | `servo_control`, motor normalization/watchdog. Mutually exclusive with serial-MCU backend, never merge. |
| `joy_control_node.py` / `joy_control_node` | Default with serial cluster | P `/action_request`, `/motor_cmd/joystick`, game mode request; S game mode state | Device scanning thread and polling timers own controller connection/heartbeats. **Exclusive joystick input device.** | remote-control config, drive mixing, neck kinematics, game hotkey/protocol. Mapping policies are mostly already delegated; lifecycle must remain Node-side. |
| `keyboard_stt_node.py` / `keyboard_stt_test_node` | One of keyboard/ASR/multimodal modes | P `voice_text` | Input thread owns stdin for test injection. | None. Deliberately simple testing adapter; no merge with real STT. |
| `llm_ros_node.py` / `walle_llm_brain` | Default except multimodal mode | P dialogue/TTS/action/BT/screen/busy/expression topics; S `voice_text`, action/BT status, game state/frame | Dedicated request worker owns one text-LLM turn at a time; invokes external LLM and camera-preview client but owns neither physical device. | LLM request/history/policy/stream/turn/tool/plan/visual/photo services, dialog workflows, game commentary and ROS adapters. Pure business flow has been extracted; remaining work is external effects and lifecycle coordination. |
| `motion_arbiter_node.py` / `motion_arbiter_node` | Default with serial cluster | P `/motor_cmd`; S joystick/tracking/autonomy motor sources and game state | Callbacks select current source and safe-stop on shutdown. Owns final motor-topic arbitration, not hardware. | `motion_arbiter`, game protocol. Safety boundary must stay independent. |
| `music_player_node.py` / `music_player_node` | Default | P music audio/spectrum/state and action status; S `/action_cmd`, game state | Owns FFmpeg/music decode session and cleanup; game entry stops playback. **Exclusive music decoder/session.** | `music_player`, music protocol, action/status protocol. Separate from shared audio-output owner. |
| `native_behavior_tree_launcher.py` / installed `wali_behavior_tree_node` | Optional `orchestration.native_behavior_tree` | Installed node: P `/behavior_tree/status`, `/action_request`; S execute/cancel/action status | `exec` handoff to native C++ plan owner; owns BT plan execution and recovery-stop lifecycle. | Registry/XML paths are injected by launcher. C++ node is a distinct failure boundary, not a Python service candidate. |
| `native_behavior_tree_ros2_bridge_launcher.py` / installed `wali_bt_ros2_bridge` | Optional with native BT | ROS 2 Action bridge (`/wali_task`) to existing BT protocol | `exec` handoff to installed bridge; owns Action goal/cancel/feedback bridge lifecycle. | No business policy in the launcher. Keep separate from native plan owner. |
| `sequence_ros_node.py` / `sequence_ros_node` | Default with serial cluster | P `/servo_cmd`, `/motor_cmd/autonomy`, `/tft_cmd`, action status; S action command/cancel, game state, tracking targets, dialog targets | 50 Hz timer applies effects; accepts cancellation and stops motion. Owns actual sequence effect scheduling, not arbitration terminal state. | `sequence_execution` library/runtime/trajectory/controller plus protocols. Service extraction is complete for this phase. |
| `serial_ros_node.py` / `walle_serial_node` | Default unless `--no-serial` | P ESP32 NETCFG response/status; S screen/TFT/music/PCA data, NETCFG request, TFT-ready | Reader and NETCFG worker threads manage one serial bridge; startup configuration sequence. **Sole ESP32/TFT serial-port owner.** | `serial_bridge`, ESP32 NETCFG services, music/RPC protocols. Must not merge with hardware bridge. |
| `stt_ros_node.py` / `walle_ear_node` | `asr_llm` mode | P `voice_text`, VAD motion; S game state, `llm_busy` | Starts/stops recognizer and mic capture according to busy/game state. **Exclusive microphone/ASR engine owner.** | `stt_service`, dialog-motion and game protocols. Keep separate from multimodal voice capture. |
| `tft_tcp_service_node.py` / `tft_tcp_service_node` | Default | P preview ready/result, game mode request; S preview request, vision command, game state/frame, music state/spectrum | Owns TFT TCP server, connected display client, preview request threads, and persistent game/tracking/music stream lifecycle. **Sole TFT TCP-port/display-stream owner.** | camera provider, TFT server/protocol, game frame/stream, music spectrum, tracking preview. Must remain separate from serial screen bridge. |
| `tts_play_node.py` / `tts_play_node` | Default | P `audio_output`; S `tts_text` | Maintains ordered TTS synthesis and paced PCM pipeline; no sound-card ownership. | TTS service/pipeline, PCM buffer/pacing/silence and turn protocol. Correct separation from audio playback. |
| `voice_chat_ros_node.py` / `voice_chat_node` | `multimodal` mode; replaces text LLM + STT | P TTS/wake/screen/action/BT/visual-search/busy/VAD/expression; S wake done, visual-search request, action/BT status, busy/game/frame | Worker/background tasks coordinate raw-audio turns; invokes external multimodal model and preview client, but does not own microphone speaker or camera. | voice-chat service, dialog turn/output/presentation/router/plan/workflows, visual search, game commentary, action/BT adapters. Business coordination has been extracted; remaining code is effect and lifecycle integration. |
| `wali_mcp_server.py` / `wali_mcp_gateway` internal ROS executor | Optional MCP | P `/action_request`; S `/action_status` | Spins a ROS executor beside MCP server; owns MCP transport/auth process lifecycle, not hardware. | `mcp_gateway`, action plan/status, ROS2 Action adapter. Preserve its security and external-client failure boundary. |
| `wali_tracking_node.py` / `wali_tracking_node` | Optional tracking | P tracking servo/motor, action status, vision command, camera command; S action command, camera status, detector output, DOA | 10 Hz control timer manages tracking activation, camera lease, loss/search/exit transitions. Does not open camera or BPU directly. | `tracking_control`, camera/vision/motion/action protocols, neck kinematics. Tracking control and loss state are already service-side; Node is effect adapter. |

## Resource ownership map

```text
microphone/ASR        stt_ros_node OR voice_chat_ros_node
sound card/mixer      audio_playback_node
TTS generation        tts_play_node
physical camera       camera_capture_node -> hobot_usb_cam
BPU detector group    hobot_vision_node
TFT TCP port/stream   tft_tcp_service_node
ESP32/TFT serial      serial_ros_node
DOA serial            doa_ros_node
I2C/PCA9685           i2c_hardware_node (ubuntu_i2c backend only)
serial-MCU output     hardware_bridge_node (serial_mcu backend only)
motor selection       motion_arbiter_node
action arbitration    action_coordinator_node
native plans          wali_behavior_tree_node
```

The map confirms that the apparently adjacent nodes have deliberately different
resource ownership, startup order, and fault containment. In particular,
camera/BPU/TFT, TTS/playback, motion arbitration/hardware output, and serial
screen/hardware bridge are not merge candidates.

## Findings and next-step decision

1. No node merge is currently justified. Every default or optional operational
   node owns a distinct lifecycle, exclusive device/process, safety boundary,
   or externally visible protocol.
2. The completed Sequence, Tracking, text LLM, and Voice Chat extractions match
   their intended boundary: their Node layers turn service decisions into ROS,
   device, and external-model effects.
3. `ai_msg_scaler_node.py` is the only visible pure-computation-heavy Node,
   but it is utility-only and bypassed by the present padded detector pipeline.
   It should be retired or refactored only when a deployment again uses it;
   changing it now would be speculative.
4. `DialogPoseSampler` is a small deterministic policy inside
   `dialog_motion_node.py`. It is a possible future service extraction only if
   dialog motion receives broader behavior changes or standalone tests; its
   current size does not justify a mechanical split.
5. Service-directory reclassification is intentionally deferred. This audit
   establishes dependency and ownership boundaries first; moving files now
   would create churn without a demonstrated import or responsibility problem.

No ROS launch, hardware node, camera, motor, or physical-device test was run
for this audit.
