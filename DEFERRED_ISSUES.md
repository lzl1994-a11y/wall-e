# Deferred Issues

This file records issues that were intentionally not fixed in this round.

## Deployment and configuration

- RDK X3 path and serial configuration are still development-machine values. Examples remain in `nodes/stt_ros_node.py`, `services/stt_service.py`, `services/serial_broker.py`, and `core/config.yaml`.
- `core/config.yaml` still contains a plaintext API key. Move it to an environment variable or an untracked local config later.
- `main.py` still imports legacy `run_llm_service`, `run_mcp_service`, `run_servo_control`, and `run_tts_service` functions that do not exist. If ROS2 nodes are now the main path, retire or rewrite `main.py` later.

## Protocol and behavior

- There is still no stable dedicated `intent` field. Current behavior uses tool calls / `action_cmd` as the action-intent carrier.
- `full_ai_text` is still published after the full LLM stream completes, so the screen AI text can lag behind streaming TTS output.
- The first-line stripping plus TTS sentence-splitting algorithm is still the old design. This round only made correction-label parsing more tolerant; a cleaner state-machine parser should be designed before changing that behavior further.
