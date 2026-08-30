"""Pure state machine for the optional game feature.

This module intentionally knows nothing about ROS, a particular emulator, the
TFT transport, or audio devices.  Nodes adapt its transitions to those systems,
which keeps the game feature removable and prevents it from becoming another
owner of robot hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GameMode(str, Enum):
    """User-visible modes for the shared controller and game session."""

    ROBOT = "robot"
    ENTERING = "entering"
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    EXITING = "exiting"


@dataclass(frozen=True)
class GamePolicy:
    """Capabilities that adapters must enforce for a mode.

    ``robot_input`` controls *all* physical robot commands, not only motor
    commands.  This avoids a game key accidentally moving a servo while the
    chassis remains stopped.
    """

    robot_input: bool
    game_input: bool
    recording: bool
    game_audio: bool
    screenshot_analysis: bool
    motors_must_stop: bool


_POLICIES = {
    GameMode.ROBOT: GamePolicy(True, False, True, False, False, False),
    GameMode.ENTERING: GamePolicy(False, False, False, False, False, True),
    GameMode.MENU: GamePolicy(False, True, False, False, False, True),
    GameMode.PLAYING: GamePolicy(False, True, False, True, True, True),
    GameMode.PAUSED: GamePolicy(False, True, False, False, False, True),
    GameMode.EXITING: GamePolicy(False, False, False, False, False, True),
}


class InvalidGameTransition(ValueError):
    """Raised when a caller requests an unsafe or impossible transition."""


class GameModeController:
    """Coordinate game lifecycle without owning any external resource.

    The caller must perform the side effects described by :attr:`policy`, then
    acknowledge them with ``game_surface_ready`` or ``robot_surface_ready``.
    This makes stopping motors and closing microphone capture prerequisites, not
    best-effort work performed after a game starts.
    """

    def __init__(self) -> None:
        self._mode = GameMode.ROBOT

    @property
    def mode(self) -> GameMode:
        return self._mode

    @property
    def policy(self) -> GamePolicy:
        return _POLICIES[self._mode]

    def request_enter(self) -> GameMode:
        """Begin a hand-controller initiated switch from robot to game mode."""
        self._require(GameMode.ROBOT)
        self._mode = GameMode.ENTERING
        return self._mode

    def game_surface_ready(self) -> GameMode:
        """Enter the ROM menu after safety and screen adapters acknowledge."""
        self._require(GameMode.ENTERING)
        self._mode = GameMode.MENU
        return self._mode

    def start_game(self) -> GameMode:
        self._require(GameMode.MENU)
        self._mode = GameMode.PLAYING
        return self._mode

    def pause_for_fault(self) -> GameMode:
        """Pause on a disconnected hand controller, TFT, or emulator fault."""
        if self._mode not in {GameMode.MENU, GameMode.PLAYING}:
            raise InvalidGameTransition(f"cannot pause from {self._mode.value}")
        self._mode = GameMode.PAUSED
        return self._mode

    def resume_game(self) -> GameMode:
        self._require(GameMode.PAUSED)
        self._mode = GameMode.PLAYING
        return self._mode

    def request_exit(self) -> GameMode:
        """Start save/teardown; robot input remains blocked until restoration."""
        if self._mode not in {
            GameMode.ENTERING,
            GameMode.MENU,
            GameMode.PLAYING,
            GameMode.PAUSED,
        }:
            raise InvalidGameTransition(f"cannot exit from {self._mode.value}")
        self._mode = GameMode.EXITING
        return self._mode

    def robot_surface_ready(self) -> GameMode:
        """Re-enable recording and robot controls after game teardown completes."""
        self._require(GameMode.EXITING)
        self._mode = GameMode.ROBOT
        return self._mode

    def _require(self, expected: GameMode) -> None:
        if self._mode is not expected:
            raise InvalidGameTransition(
                f"expected {expected.value}, current mode is {self._mode.value}"
            )


__all__ = [
    "GameMode",
    "GameModeController",
    "GamePolicy",
    "InvalidGameTransition",
]
