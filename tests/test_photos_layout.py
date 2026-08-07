"""Regression tests for the display-scale window layout.

Guards the capture-layout mistakes that hit production:
1. A DPI-unaware process reads display-scaled system metrics (1707x1067 for
   a 2560x1600 screen at 150%), which made 6 pinned windows impossible.
   The module must set DPI awareness at import so layout math runs in
   physical pixels.
2. ``_page_display_rects`` must never hand out overlapping rects: when the
   requested zoom cannot fit the usable area it shrinks uniformly (auto-fit,
   down to a 12% floor) and only then drops rows -- and the batch threads
   must match the effective window count so two windows never share a rect.
3. The grid hugs the usable area edges (top row to the top, last row to
   the bottom, left window to the left, rightmost window to the right) with
   a small minimum gap, and respects ``right_limit`` so the rightmost
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


def _assert_disjoint(testcase, rects, keys):
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            testcase.assertEqual(
                _overlap(rects[keys[i]], rects[keys[j]]), 0,
                f"{rects[keys[i]]} overlaps {rects[keys[j]]}",
            )


class DpiAwarenessTests(unittest.TestCase):
    def test_process_is_dpi_aware(self):
        """Layout runs in physical pixels, never display-scaled metrics."""
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            shcore = ctypes.windll.shcore
            value = ctypes.c_int(0)
            # Two-arg call (process handle + out param): the one-arg form
            # raises E_INVALIDARG and crashes the interpreter at exit.
            handle = kernel32.OpenProcess(
                0x0400,  # PROCESS_QUERY_INFORMATION
                False,
                kernel32.GetCurrentProcessId(),
            )
            try:
                result = shcore.GetProcessDpiAwareness(handle, ctypes.byref(value))
            finally:
                kernel32.CloseHandle(handle)
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
        # two rows fit and all six requested windows stay active at the
        # requested zoom.
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count, fraction = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 2560, 1600)
        )
        self.assertEqual(count, 6)
        self.assertEqual(fraction, 0.25)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        _assert_disjoint(self, rects, keys)
        # Top row hugs the top, bottom row hugs the bottom.
        self.assertEqual(concurrent[0][1], 24)
        self.assertLessEqual(concurrent[3][3], 1600)

    def test_tall_pages_shrink_to_fit_six_windows_on_short_screen(self):
        # The same 720px windows cannot stack on a 1707x1067 screen at the
        # requested zoom: auto-fit shrinks the zoom (alignment kept, windows
        # equal), so all six windows still run instead of dropping rows.
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count, fraction = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 1707, 1067)
        )
        self.assertEqual(count, 6)
        self.assertLess(fraction, 0.25)
        keys = [str(index) for index in range(1, 7)]
        _assert_disjoint(self, rects, keys)
        for rect in rects.values():
            self.assertLessEqual(rect[2], 1707)
            self.assertLessEqual(rect[3], 1067)
        self.assertEqual(rects["1"][1], 24)
        self.assertEqual(rects["4"][3], 1067 - 24)

    def test_rows_dropped_only_when_floor_cannot_fit(self):
        # 2005x5000 pages: even at the 12% floor two rows do not fit on a
        # 1707x1067 screen, so rows are dropped to a single row of 3 at the
        # full requested zoom -- still disjoint and inside the screen.
        pages = _make_pages(self.directory, (2005, 5000), count=6)
        rects, count, fraction = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 1707, 1067)
        )
        self.assertEqual(count, 3)
        self.assertEqual(fraction, 0.25)
        keys = [str(index) for index in range(1, 4)]
        _assert_disjoint(self, rects, keys)
        for rect in rects.values():
            self.assertLessEqual(rect[2], 1707)
            self.assertLessEqual(rect[3], 1067)
        self.assertEqual(rects["1"][1], 24)

    def test_user_scenario_2020x2880_42pct_region_2500x1400(self):
        # The production case that hit overlap: 2020x2880 pages at 42% need
        # two ~1210px rows, which never fit a 2500x1400 region.  Auto-fit
        # shrinks to ~23% so all six windows run, edge-aligned and disjoint,
        # with no window past the 2400 right limit.
        pages = _make_pages(self.directory, (2020, 2880), count=6)
        rects, count, fraction = _page_display_rects(
            pages,
            42,
            parallel=6,
            screen_bounds=(0, 0, 2500, 1400),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        self.assertLess(fraction, 0.30)
        self.assertGreater(fraction, 0.20)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        _assert_disjoint(self, rects, keys)
        for rect in concurrent:
            self.assertLessEqual(rect[2], 2400)
            self.assertLessEqual(rect[3], 1400)
        # Equal page sizes -> equal windows, aligned grid.
        sizes = {(r[2] - r[0], r[3] - r[1]) for r in concurrent}
        self.assertEqual(len(sizes), 1)
        self.assertEqual(concurrent[0][1], 24)
        self.assertEqual(concurrent[3][3], 1400 - 24)
        # The auto-fit budget keeps the full minimum gap between the rows:
        # a drifted window can never intrude into the neighbouring row.
        self.assertGreaterEqual(concurrent[3][1] - concurrent[0][3], 32)
        self.assertGreaterEqual(concurrent[1][0] - concurrent[0][2], 32)

    def test_small_pages_keep_six_windows(self):
        # 985x1400 pages at 25% = ~320x350 windows (min_width floor): two
        # rows fit everywhere.
        pages = _make_pages(self.directory, (985, 1400), count=6)
        rects, count, fraction = _page_display_rects(
            pages, 25, parallel=6, screen_bounds=(0, 0, 1707, 1067)
        )
        self.assertEqual(count, 6)
        self.assertEqual(fraction, 0.25)

    def test_wide_pages_shrink_for_horizontal_fit(self):
        # Wide 2000x1000 pages at 42%: three 840px columns cannot fit
        # 2352px of usable width, so auto-fit shrinks until the row fits;
        # all six windows stay and none crosses the right limit.
        pages = _make_pages(self.directory, (2000, 1000), count=6)
        rects, count, fraction = _page_display_rects(
            pages,
            42,
            parallel=6,
            screen_bounds=(0, 0, 2500, 1400),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        self.assertLess(fraction, 0.42)
        keys = [str(index) for index in range(1, 7)]
        _assert_disjoint(self, rects, keys)
        for rect in rects.values():
            self.assertLessEqual(rect[2], 2400)
            self.assertLessEqual(rect[3], 1400)

    def test_any_parallel_count_is_safe(self):
        # parallel=1..5 (GUI range is 1..12): every concurrent window gets
        # a distinct, disjoint tile; auto-fit keeps all requested windows
        # inside the region, whatever the parallel count.
        pages = _make_pages(self.directory, (2020, 2880), count=6)
        for parallel in range(1, 6):
            rects, count, fraction = _page_display_rects(
                pages[:parallel], 42, parallel=parallel,
                screen_bounds=(0, 0, 2500, 1400), right_limit=2400,
            )
            self.assertEqual(count, parallel)
            keys = [str(index) for index in range(1, parallel + 1)]
            _assert_disjoint(self, rects, keys)
            for key in keys:
                self.assertLessEqual(rects[key][2], 2400)
                self.assertLessEqual(rects[key][3], 1400)

    def test_right_limit_keeps_windows_left_of_boundary(self):
        pages = _make_pages(self.directory, (2005, 2880), count=6)
        rects, count, fraction = _page_display_rects(
            pages,
            25,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        self.assertEqual(fraction, 0.25)
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
        rects, count, fraction = _page_display_rects(
            pages,
            31,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
            row_gap_px=20,
            bottom_shift_up=40,
        )
        self.assertEqual(count, 6)
        self.assertEqual(fraction, 0.31)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        _assert_disjoint(self, rects, keys)
        for i in range(6):
            x1, y1, x2, y2 = concurrent[i]
            self.assertEqual((x2 - x1, y2 - y1), (744, 496))
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
        rects, count, fraction = _page_display_rects(
            pages,
            42,
            parallel=6,
            screen_bounds=(0, 0, 2560, 1600),
            right_limit=2400,
        )
        self.assertEqual(count, 6)
        self.assertEqual(fraction, 0.42)
        keys = [str(index) for index in range(1, 7)]
        concurrent = [rects[key] for key in keys]
        _assert_disjoint(self, rects, keys)
        for i in range(6):
            x1, y1, x2, y2 = concurrent[i]
            self.assertEqual((x2 - x1, y2 - y1), (538, 752))
        self.assertLessEqual(max(rect[2] for rect in concurrent), 2400)
        self.assertLessEqual(max(rect[3] for rect in concurrent), 1600)


if __name__ == "__main__":
    unittest.main()
