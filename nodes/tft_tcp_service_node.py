#!/usr/bin/env python3
"""Own the chest TFT TCP service and route camera, game, and music frames."""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray

from services.camera_frame import CameraFrameProvider
from services.game_frame_adapter import GameFrameAdapter
from services.game_protocol import (
    GAME_FRAME_TOPIC,
    GAME_MODE_REQUEST_TOPIC,
    GAME_MODE_STATE_TOPIC,
    GAME_SURFACE_READY,
    decode_game_frame,
    encode_game_request,
    game_mode_from_message,
)
from services.game_tft_stream import GameTftStreamServer
from services.music_protocol import (
    MUSIC_SPECTRUM_TOPIC,
    MUSIC_SPECTRUM_FPS,
    MUSIC_STATE_TOPIC,
    decode_music_state,
)
from services.music_spectrum import render_spectrum_frame
from services.tft_preview_protocol import (
    TFT_PREVIEW_READY_TOPIC,
    TFT_PREVIEW_REQUEST_TOPIC,
    TFT_PREVIEW_RESULT_TOPIC,
    decode_preview_request,
    encode_preview_result,
)
from services.tft_preview_server import PreviewResult, load_tft_preview_settings
from services.tracking_tft_preview import TrackingTftPreview
from services.vision_pipeline_protocol import (
    VISION_PIPELINE_COMMAND_TOPIC,
    decode_vision_pipeline_command,
)


class TftTcpServiceNode(Node):
    """The only process allowed to bind the TFT TCP port."""

    def __init__(self) -> None:
        super().__init__("tft_tcp_service_node")
        self.settings = load_tft_preview_settings()
        self.server = GameTftStreamServer(self.settings, logger=self.get_logger())
        self.camera_frames = CameraFrameProvider(self)
        self.tracking_preview = TrackingTftPreview(
            self.server,
            self.camera_frames,
            fps=self.settings.fps,
            logger=self.get_logger(),
        )
        self._game_mode = "robot"
        self._game_stream = None
        self._game_frame_adapter = None
        self._tracking_was_enabled = False
        self._music_state = "stopped"
        self._music_stream = None
        self._music_frame_adapter = None
        self._music_tracking_was_enabled = False
        self._stop_event = threading.Event()
        self._preview_threads: set[threading.Thread] = set()
        self._preview_threads_lock = threading.Lock()

        self._ready_publisher = self.create_publisher(
            String, TFT_PREVIEW_READY_TOPIC, 10
        )
        self._result_publisher = self.create_publisher(
            String, TFT_PREVIEW_RESULT_TOPIC, 10
        )
        self._game_request_publisher = self.create_publisher(
            String, GAME_MODE_REQUEST_TOPIC, 10
        )
        self.create_subscription(
            String, TFT_PREVIEW_REQUEST_TOPIC, self._on_preview_request, 10
        )
        self.create_subscription(
            String,
            VISION_PIPELINE_COMMAND_TOPIC,
            self._on_vision_pipeline_command,
            10,
        )
        self.create_subscription(String, GAME_MODE_STATE_TOPIC, self._on_game_state, 10)
        self.create_subscription(UInt8MultiArray, GAME_FRAME_TOPIC, self._on_game_frame, 1)
        self.create_subscription(String, MUSIC_STATE_TOPIC, self._on_music_state, 10)
        self.create_subscription(
            Float32MultiArray, MUSIC_SPECTRUM_TOPIC, self._on_music_spectrum, 1
        )

        self._listening = False
        try:
            self.server.start()
            self._listening = True
            self._publish_ready()
            self._ready_timer = self.create_timer(1.0, self._publish_ready)
        except Exception as exc:
            self.get_logger().error(f"TFT TCP service failed to start: {exc}")

    def _publish_ready(self) -> None:
        if self._listening:
            self._ready_publisher.publish(String(data="ready"))

    def _on_vision_pipeline_command(self, message) -> None:
        command = decode_vision_pipeline_command(message.data)
        if command is not None:
            self.tracking_preview.set_command(command)

    def _on_game_state(self, message) -> None:
        mode = game_mode_from_message(message.data)
        if mode is None:
            return
        previous = self._game_mode
        self._game_mode = mode
        if mode != "robot":
            if previous == "robot":
                music_was_tracking = self._music_tracking_was_enabled
                self._close_music_stream(resume_tracking=False)
                self._tracking_was_enabled = (
                    music_was_tracking or self.tracking_preview.pause()
                )
            self._ensure_game_stream()
            return
        if previous == "robot":
            return
        self._close_game_stream()
        if self._music_state == "playing":
            self._ensure_music_stream(
                tracking_was_enabled=self._tracking_was_enabled
            )
        elif self._tracking_was_enabled:
            self.tracking_preview.resume()
        self._tracking_was_enabled = False

    def _ensure_game_stream(self) -> None:
        if self._game_frame_adapter is not None:
            return
        stream = self.server.open_jpeg_stream(fps=10)
        if stream is None:
            self.get_logger().warning("游戏 TFT 流暂不可用")
            return
        self._game_stream = stream
        self._game_frame_adapter = GameFrameAdapter(stream, fps=10)
        self._game_request_publisher.publish(
            String(data=encode_game_request(GAME_SURFACE_READY))
        )

    def _close_game_stream(self) -> None:
        adapter = self._game_frame_adapter
        self._game_frame_adapter = None
        if adapter is not None:
            adapter.close()
        stream = self._game_stream
        self._game_stream = None
        if stream is not None:
            stream.close()

    def _on_game_frame(self, message) -> None:
        if self._game_mode == "robot":
            return
        frame = decode_game_frame(bytes(message.data))
        if frame is None:
            return
        adapter = self._game_frame_adapter
        if adapter is not None:
            adapter.submit_frame(*frame)

    def _on_music_state(self, message) -> None:
        state = decode_music_state(message.data)
        if state is None:
            return
        self._music_state = state["state"]
        if self._music_state == "playing":
            self._ensure_music_stream()
        elif self._music_state in {"stopped", "error"}:
            self._close_music_stream()

    def _ensure_music_stream(self, *, tracking_was_enabled: bool | None = None) -> None:
        if self._game_mode != "robot" or self._music_frame_adapter is not None:
            return
        self._music_tracking_was_enabled = (
            self.tracking_preview.pause()
            if tracking_was_enabled is None else tracking_was_enabled
        )
        stream = self.server.open_jpeg_stream(fps=MUSIC_SPECTRUM_FPS)
        if stream is None:
            if self._music_tracking_was_enabled:
                self.tracking_preview.resume()
            self._music_tracking_was_enabled = False
            self.get_logger().warning("音乐频谱 TFT 流暂不可用")
            return
        self._music_stream = stream
        self._music_frame_adapter = GameFrameAdapter(
            stream, fps=MUSIC_SPECTRUM_FPS
        )

    def _close_music_stream(self, *, resume_tracking: bool = True) -> None:
        adapter, self._music_frame_adapter = self._music_frame_adapter, None
        if adapter is not None:
            adapter.close()
        stream, self._music_stream = self._music_stream, None
        if stream is not None:
            stream.close()
        if resume_tracking and self._game_mode == "robot" and self._music_tracking_was_enabled:
            self.tracking_preview.resume()
        self._music_tracking_was_enabled = False

    def _on_music_spectrum(self, message) -> None:
        adapter = self._music_frame_adapter
        if self._game_mode != "robot" or self._music_state != "playing" or adapter is None:
            return
        adapter.submit_frame(*render_spectrum_frame(message.data))

    def _on_preview_request(self, message) -> None:
        request = decode_preview_request(message.data)
        if request is None:
            self.get_logger().warning("忽略无效 TFT 预览请求")
            return
        thread = threading.Thread(
            target=self._run_preview_request,
            args=(request,),
            name=f"tft-preview-{request['request_id'][:8]}",
            daemon=True,
        )
        with self._preview_threads_lock:
            self._preview_threads.add(thread)
        thread.start()

    def _run_preview_request(self, request: dict) -> None:
        try:
            if self._stop_event.is_set():
                result = PreviewResult(error="tft_preview_shutting_down")
            else:
                music_was_active = self._music_state == "playing"
                music_was_tracking = self._music_tracking_was_enabled
                if music_was_active:
                    self._close_music_stream(resume_tracking=False)
                was_tracking = music_was_tracking or self.tracking_preview.pause()
                try:
                    result = self.server.send_camera_preview(
                        self.camera_frames,
                        duration_ms=request["duration_ms"],
                        hold_ms=request["hold_ms"],
                        fps=request["fps"],
                        should_stop=self._stop_event.is_set,
                    )
                finally:
                    if (
                        music_was_active
                        and self._music_state == "playing"
                        and not self._stop_event.is_set()
                    ):
                        self._ensure_music_stream(tracking_was_enabled=was_tracking)
                    elif was_tracking and not self._stop_event.is_set():
                        self.tracking_preview.resume()
            self._result_publisher.publish(
                String(data=encode_preview_result(request["request_id"], result))
            )
        except Exception as exc:
            self.get_logger().error(f"TFT preview request failed: {exc}")
            result = PreviewResult(error=str(exc))
            self._result_publisher.publish(
                String(data=encode_preview_result(request["request_id"], result))
            )
        finally:
            with self._preview_threads_lock:
                self._preview_threads.discard(threading.current_thread())

    def destroy_node(self):
        self._stop_event.set()
        self._close_game_stream()
        self._close_music_stream(resume_tracking=False)
        self.tracking_preview.stop()
        self.server.stop()
        with self._preview_threads_lock:
            threads = list(self._preview_threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TftTcpServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
