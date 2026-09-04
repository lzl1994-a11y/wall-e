import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from services import fc_game_session as session_module


class _Sink:
    def set_key(self, _key, _down):
        pass

    def close(self):
        pass


class _Relay:
    instances = []

    def __init__(self, _path, sink):
        self.sink = sink
        self.error = None
        self.__class__.instances.append(self)

    def start(self):
        pass

    def switch_sink(self, sink):
        previous = self.sink
        self.sink = sink
        previous.close()

    def stop(self):
        self.sink.close()


class _Core:
    instances = []

    def __init__(self, _path, *, on_frame, audio_sink):
        self.joypad = _Sink()
        self.audio_sink = audio_sink
        self.closed = False
        self.__class__.instances.append(self)

    def load(self, _rom):
        pass

    def run_until(self, _should_stop):
        pass

    def close(self):
        self.closed = True


class _Menu(_Sink):
    instances = []
    rom = Path("mario.nes")
    stop = None

    def __init__(self, _roms, *, on_frame):
        self.chosen = self.rom if not self.__class__.instances else None
        self.__class__.instances.append(self)

    def emit(self):
        if len(self.__class__.instances) == 2:
            self.stop.set()


class _Playback:
    def __init__(self):
        self.turn_ends = 0

    def play(self, _samples):
        pass

    def mark_turn_end(self):
        self.turn_ends += 1


class FcGameSessionTests(unittest.TestCase):
    def setUp(self):
        _Relay.instances = []
        _Core.instances = []
        _Menu.instances = []

    def test_return_to_menu_closes_per_game_resources_and_notifies_state(self):
        stop = threading.Event()
        _Menu.stop = stop
        playback = _Playback()
        events = []
        session = session_module.FcGameSession(
            core_path="core.so",
            rom_directory="roms",
            controller_path="event2",
            on_frame=lambda *_args: None,
            playback=playback,
            on_game_started=lambda rom: events.append(("started", rom)),
            on_return_to_menu=lambda: events.append(("menu", None)),
            on_audio_end_queued=lambda: events.append(("audio_end", None)),
        )

        with (
            patch.object(session_module, "discover_roms", return_value=[_Menu.rom]),
            patch.object(session_module, "GameMenu", _Menu),
            patch.object(session_module, "FcControllerRelay", _Relay),
            patch.object(session_module, "LibretroFc", _Core),
        ):
            session.run(stop)

        self.assertEqual(events, [
            ("started", _Menu.rom),
            ("menu", None),
            ("audio_end", None),
        ])
        self.assertTrue(_Core.instances[0].closed)
        self.assertEqual(playback.turn_ends, 1)
        self.assertIs(_Relay.instances[0].sink, _Menu.instances[1])


if __name__ == "__main__":
    unittest.main()
