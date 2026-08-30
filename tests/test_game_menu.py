import tempfile
import unittest
from pathlib import Path

from services.game_menu import GameMenu, discover_roms


class GameMenuTests(unittest.TestCase):
    def test_discovers_nes_case_insensitively_and_sorts_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.NES").touch()
            (root / "A.nes").touch()
            (root / "ignore.txt").touch()

            self.assertEqual(
                [path.name for path in discover_roms(root)],
                ["A.nes", "z.NES"],
            )

    def test_direction_wraps_and_a_chooses(self):
        roms = [Path("one.nes"), Path("two.NES")]
        menu = GameMenu(roms)

        menu.set_key("KP_8", True)
        menu.set_key("KP_8", False)
        self.assertEqual(menu.selected, 1)
        menu.set_key("F", True)

        self.assertEqual(menu.chosen, roms[1])

    def test_render_is_a_raw_256_by_240_bgr_frame(self):
        frame = GameMenu([Path("Bomber Man (J).nes")]).render()

        self.assertEqual(frame.shape, (240, 256, 3))
        self.assertEqual(frame.dtype.name, "uint8")


if __name__ == "__main__":
    unittest.main()
