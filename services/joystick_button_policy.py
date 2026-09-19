"""Pure decision service for joystick buttons and directional hat events."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

from services.game_hotkey import ButtonChordHold

HAT_X = 16
HAT_Y = 17

BTN_A = 304
BTN_B = 305
BTN_X = 307
BTN_Y = 308
BTN_L1 = 310
BTN_R1 = 311

DEFAULT_HOLD_SECONDS = 2.0
DEFAULT_AUTO_RESET_DELAY = 3.0


@dataclass(frozen=True)
class JoystickButtonDecision:
    """Decision produced by a discrete button or hat event."""

    action: str | None = None
    timer_updates: dict[str, float] = field(default_factory=dict)


class JoystickButtonPolicy:
    """Tracks discrete button states and maps button/hat events to robot actions and timers."""

    def __init__(
        self,
        *,
        hold_seconds: float = DEFAULT_HOLD_SECONDS,
        chord_clock: Callable[[], float] | None = None,
    ):
        if chord_clock is not None:
            self._chord_hotkey = ButtonChordHold(hold_seconds=hold_seconds, clock=chord_clock)
        else:
            self._chord_hotkey = ButtonChordHold(hold_seconds=hold_seconds)

        self._chord_fired = False
        self._x_down = False
        self._y_down = False

    @property
    def x_down(self) -> bool:
        return self._x_down

    @property
    def y_down(self) -> bool:
        return self._y_down

    @property
    def chord_fired(self) -> bool:
        return self._chord_fired

    def poll_game_toggle(self, *, game_active: bool = False) -> bool:
        """Poll the chord hotkey (X + Y held for 2s) in non-game mode.

        Returns True exactly once when the chord criteria is met.
        """
        if not game_active and self._chord_hotkey.poll():
            self._chord_fired = True
            return True
        return False

    def handle_key(
        self,
        code: int,
        value: int,
        *,
        now: float = 0.0,
        auto_reset_delay: float = DEFAULT_AUTO_RESET_DELAY,
        game_active: bool = False,
    ) -> JoystickButtonDecision:
        """Handle EV_KEY discrete button events."""
        # 1. X / Y button handling
        if code in {BTN_X, BTN_Y}:
            if value not in {0, 1}:
                return JoystickButtonDecision()

            down = value == 1
            if code == BTN_X:
                self._x_down = down
                self._chord_hotkey.set_first(down)
                other_down = self._y_down
            else:
                self._y_down = down
                self._chord_hotkey.set_second(down)
                other_down = self._x_down

            action: str | None = None
            if (
                not down
                and not other_down
                and not self._chord_fired
                and not game_active
            ):
                if code == BTN_X:
                    action = "wave_hello"
                else:
                    action = "raise_hand"

            if not self._x_down and not self._y_down:
                self._chord_fired = False

            return JoystickButtonDecision(action=action)

        # 2. Other buttons suppressed in game mode
        if game_active:
            return JoystickButtonDecision()

        # 3. Non-game mode: only value == 1 (press) triggers actions/timers
        if value != 1:
            return JoystickButtonDecision()

        if code == BTN_L1:
            return JoystickButtonDecision(
                timer_updates={"eyebrow_l": now + auto_reset_delay}
            )
        if code == BTN_R1:
            return JoystickButtonDecision(
                timer_updates={"eyebrow_r": now + auto_reset_delay}
            )
        if code == BTN_A:
            return JoystickButtonDecision(action="happy_dance")
        if code == BTN_B:
            return JoystickButtonDecision(action="sad_react")

        return JoystickButtonDecision()

    def handle_hat(
        self,
        code: int,
        value: int,
        *,
        now: float = 0.0,
        auto_reset_delay: float = DEFAULT_AUTO_RESET_DELAY,
    ) -> JoystickButtonDecision:
        """Handle EV_ABS directional hat events.

        Hat events are not suppressed by game mode.
        """
        timer_updates: dict[str, float] = {}
        if code == HAT_X:
            if value == -1:  # Left
                timer_updates["arm_l"] = now + auto_reset_delay
            elif value == 1:  # Right
                timer_updates["arm_r"] = now + auto_reset_delay
        elif code == HAT_Y:
            if value == -1:  # Up
                timer_updates["arm_l"] = now + auto_reset_delay
                timer_updates["arm_r"] = now + auto_reset_delay
            elif value == 1:  # Down
                timer_updates["arm_l"] = 0.0
                timer_updates["arm_r"] = 0.0

        return JoystickButtonDecision(timer_updates=timer_updates)

__all__ = [
    "HAT_X",
    "HAT_Y",
    "BTN_A",
    "BTN_B",
    "BTN_X",
    "BTN_Y",
    "BTN_L1",
    "BTN_R1",
    "DEFAULT_HOLD_SECONDS",
    "DEFAULT_AUTO_RESET_DELAY",
    "JoystickButtonDecision",
    "JoystickButtonPolicy",
]
