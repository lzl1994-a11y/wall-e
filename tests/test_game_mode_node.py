import importlib
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy

from services.game.game_exit_barrier import GameExitBarrier
from services.game.game_mode import GameModeController


class _FakeMsg:
    def __init__(self, data=""):
        self.data = data


def import_game_mode_node():
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy_node = types.ModuleType("rclpy.node")
    fake_std_msgs = types.ModuleType("std_msgs")
    fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
    fake_rclpy_node.Node = object
    fake_std_msgs_msg.String = _FakeMsg
    fake_std_msgs_msg.UInt8MultiArray = _FakeMsg

    sys.modules.pop("nodes.game_mode_node", None)
    with patch.dict(
        sys.modules,
        {
            "rclpy": fake_rclpy,
            "rclpy.node": fake_rclpy_node,
            "std_msgs": fake_std_msgs,
            "std_msgs.msg": fake_std_msgs_msg,
        },
    ):
        return importlib.import_module("nodes.game_mode_node")


game_mode_node = import_game_mode_node()


class GameModeNodeBoundaryTests(unittest.TestCase):
    def make_node(self):
        node = game_mode_node.GameModeNode.__new__(game_mode_node.GameModeNode)
        node._controller = GameModeController()
        node._state_pub = Mock()
        node._frame_pub = Mock()
        node._audio_pub = Mock()
        node._playback = game_mode_node._RosPlaybackSink(node._audio_pub)
        node._session = None
        node._session_thread = None
        node._session_stop = threading.Event()
        node._controller_path = "/dev/input/event2"
        node._state_lock = threading.RLock()
        node._frame_lock = threading.Lock()
        node._latest_frame = None
        node._frame_sequence = 0
        node._published_frame_sequence = 0
        node._exit_barrier = GameExitBarrier()
        node._exit_timer = None
        node.get_logger = Mock()
        return node

    def enter_menu(self, node):
        node._controller.request_enter()
        node._controller.game_surface_ready()

    def test_session_finish_without_trailing_audio_completes_immediately(self):
        node = self.make_node()
        self.enter_menu(node)

        node._finish_session()

        self.assertEqual(node._controller.mode.value, "robot")
        self.assertIsNone(node._exit_timer)

    def test_session_finish_awaiting_audio_starts_timeout_without_immediate_restore(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        with patch("threading.Timer") as mock_timer_cls:
            mock_timer = Mock()
            mock_timer_cls.return_value = mock_timer

            node._finish_session()

            self.assertEqual(node._controller.mode.value, "exiting")
            mock_timer_cls.assert_called_once_with(15.0, node._complete_exit)
            mock_timer.start.assert_called_once()
            self.assertIs(node._exit_timer, mock_timer)

    def test_session_finished_then_idle_completes_exit_and_cancels_timer(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        mock_timer = Mock()
        with patch("threading.Timer", return_value=mock_timer):
            node._finish_session()

        self.assertEqual(node._controller.mode.value, "exiting")
        node._on_llm_busy(_FakeMsg("idle"))

        self.assertEqual(node._controller.mode.value, "robot")
        mock_timer.cancel.assert_called_once()
        self.assertIsNone(node._exit_timer)

    def test_audio_finished_first_does_not_prematurely_exit_and_session_finish_restores_robot(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        node._on_llm_busy(_FakeMsg("idle"))
        self.assertEqual(node._controller.mode.value, "menu")

        node._finish_session()
        self.assertEqual(node._controller.mode.value, "robot")
        self.assertIsNone(node._exit_timer)

    def test_timeout_fires_and_recovers_robot(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        with patch("threading.Timer", return_value=Mock()):
            node._finish_session()

        self.assertEqual(node._controller.mode.value, "exiting")
        node._complete_exit()
        self.assertEqual(node._controller.mode.value, "robot")

    def test_complete_exit_when_not_exiting_is_noop_and_preserves_barrier(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        # Calling _complete_exit while still in menu
        node._complete_exit()
        self.assertEqual(node._controller.mode.value, "menu")

        # Barrier state must NOT have been reset by the no-op _complete_exit()
        with patch("threading.Timer", return_value=Mock()):
            node._finish_session()
        self.assertEqual(node._controller.mode.value, "exiting")

    def test_busy_only_mutes_playback_and_does_not_mark_audio_finished(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        node._on_llm_busy(_FakeMsg("busy"))
        self.assertTrue(node._playback.muted)

        with patch("threading.Timer", return_value=Mock()):
            node._finish_session()

        self.assertEqual(node._controller.mode.value, "exiting")

    def test_barrier_reset_after_exit_ready_for_subsequent_round(self):
        node = self.make_node()
        self.enter_menu(node)
        node._prepare_audio_end()

        mock_timer = Mock()
        with patch("threading.Timer", return_value=mock_timer):
            node._finish_session()
            node._on_llm_busy(_FakeMsg("idle"))

        self.assertEqual(node._controller.mode.value, "robot")

        # Second round: without trailing audio, finishes immediately
        self.enter_menu(node)
        node._finish_session()
        self.assertEqual(node._controller.mode.value, "robot")
        self.assertIsNone(node._exit_timer)


if __name__ == "__main__":
    unittest.main()
