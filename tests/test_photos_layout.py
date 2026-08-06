"""Regression tests for the display-scale window layout.

Guards the two capture-layout mistakes that hit production:
1. A DPI-unaware process reads display-scaled system metrics (1707x1067 for
   a 2560x1600 screen at 150%), which made 6 pinned windows impossible.
   The module must set DPI awareness at import so layout math runs in
   physical pixels.
2. ``_page_display_rects`` must drop rows that cannot fit (two 720px rows on
   a short screen overlap) and must respect ``right_limit`` so the rightmost
   window never crosses a configured boundary.
"""

import ctypes
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photos_capture import _page_display_rects


def _make_pages(directory: Path, size, count: int = 8):
    paths = []
    for index in range(1, count + 1):
        path = directory / f"{index}.png"
        Image.new("RGB", size, (255, 255, 255)).save(path)
        paths.append(path)
    return paths


def _overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


class DpiAwarenessTests(unittest.TestCase):
    def test_process_is_dpi_aware(self):
        """Layout runs in physical pixels, never display-scaled metrics."""
        try:
            value = ctypes.c_int(0)
            result = ctypes.windll.shcore.GetProcessDpiAwareness(ctypes.byref(value))
        except Exception:
            result = -1
        if result == 0:  # S_OK
            # 1 = system DPI aware, 2 = per-monitor DPI aware; 0 would mean
            # the 1707x1067-vs-2560x1600 bug is back.
            self.assertIn(value.value, (1, 2))
        else:
            try:
                aware = ctypes.windll.user32.IsProcessDPIAware()
            except Exception:
                self.skipTest("DPI APIs unavailable")
            self.assertEqual(aware, 1)


class DisplayScaleLayoutTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="photos_layout_")
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def test_tall_pages_fit_two_rows_on_big_screen(self):
        # 2005x2880 pages at 25% = ~501x720 windows; on a 2560x1600 screen
        # two rows fit and all six requested windows stay active.
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 2560, 1600)
        )
        self.assertEqual(count, 6)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        for i in range(6):
            for j in range(i + 1, 6):
                self.assertEqual(_overlap(concurrent[i], concurrent[j]), 0)
        # Top row hugs the top, bottom row hugs the bottom.
        self.assertEqual(concurrent[0][1], 24)
        self.assertLessEqual(concurrent[3][3], 1600)

    def test_tall_pages_fall_back_to_single_row_on_short_screen(self):
        # The same 720px windows cannot stack on a 1707x1067 screen: rows
        # are dropped (zoom kept, alignment kept), so only 3 run at once.
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 1707, 1067)
        )
        self.assertEqual(count, 3)
        keys = [str(index) for index in range(1, 4)]
        concurrent = [rects[key] for key in keys]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertEqual(_overlap(concurrent[i], concurrent[j]), 0)

    def test_small_pages_keep_six_windows(self):
        # 985x1400 pages at 25% = ~320x350 windows: two rows fit everywhere.
        pages = _make_pages(self.directory, (985, 1400), count=6)
        rects, count = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 1707, 1067)
        )
        self.assertEqual(count, 6)

    def test_right_limit_keeps_windows_left_of_boundary(self):
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count = _page_display_rects(
            pages,
            25,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        for rect in rects.values():
            self.assertLessEqual(rect[2], 2400)
        # The rightmost window still hugs the boundary (gap rounding keeps
        # it a few px short of the exact limit).
        right_edges = [rect[2] for rect in rects.values()]
        self.assertGreaterEqual(max(right_edges), 2400 - 50)

    def test_fixed_row_gap_insets_rightmost_windows(self):
        # 2400x1600 pages at 31% = 744x496 windows; with a fixed 20px
        # column gap the rightmost window moves left of the spread layout,
        # and the second row lifts 40px off the bottom edge.
        pages = _make_pages(self.directory, (2400, 1600), count=6)
        rects, count = _page_display_rects(
            pages,
            31,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
            row_gap_px=20,
            bottom_shift_up=40,
        )
        self.assertEqual(count, 6)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        for i in range(6):
            x1, y1, x2, y2 = concurrent[i]
            self.assertEqual((x2 - x1, y2 - y1), (744, 496))
            for j in range(i + 1, 6):
                self.assertEqual(_overlap(concurrent[i], concurrent[j]), 0)
        # Columns at x = 24, 788, 1552 -> rightmost edge 2296 (20px gaps).
        right_edges = sorted(rect[2] for rect in concurrent)
        self.assertEqual(right_edges[0], 768)
        self.assertEqual(right_edges[4], 2296)
        self.assertEqual(right_edges[5], 2296)
        # Top row hugs the top, second row sits 40px off the bottom.
        self.assertEqual(concurrent[0][1], 24)
        self.assertEqual(max(rect[1] for rect in concurrent), 1600 - 24 - 496 - 40)

    def test_portrait_pages_keep_six_windows(self):
        # 1280x1790 pages at 42% = 538x752 windows: 3x2 fits on a
        # 2560x1600 screen with no overlap.
        pages = _make_pages(self.directory, (1280, 1790), count=6)
        rects, count = _page_display_rects(
            pages,
            42,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        for i in range(6):
            x1, y1, x2, y2 = concurrent[i]
            self.assertEqual((x2 - x1, y2 - y1), (538, 752))
            for j in range(i + 1, 6):
                self.assertEqual(_overlap(concurrent[i], concurrent[j]), 0)
        self.assertLessEqual(max(rect[2] for rect in concurrent), 2400)
        self.assertLessEqual(max(rect[3] for rect in concurrent), 1600)


if __name__ == "__main__":
    unittest.main()
