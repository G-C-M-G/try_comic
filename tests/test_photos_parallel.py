import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from photos_capture import (
    _ensure_window_rect,
    _tile_rects,
    capture_photos_fast_batch,
)


def _fake_image(path: Path) -> Path:
    Image.new("RGB", (100, 140), (200, 200, 200)).save(path)
    return path


def _ok_manifest(frames_only=False):
    return {
        "schema_version": 2,
        "backend": "Windows Photos Scan Text",
        "source": {"path": "x", "width": 100, "height": 140},
        "capture": {
            "complete": True,
            "frames_only": frames_only,
            "selection_blocks": [],
            "timing": {"foreground_active_seconds": 2.0},
        },
        "columns": [],
    }


class TileRectTests(unittest.TestCase):
    def test_tiles_do_not_overlap(self):
        rects = _tile_rects(3)
        self.assertEqual(len(rects), 3)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                ax1, ay1, ax2, ay2 = rects[i]
                bx1, by1, bx2, by2 = rects[j]
                overlap = (
                    ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2
                )
                self.assertFalse(overlap, f"{rects[i]} overlaps {rects[j]}")

    def test_tiles_are_sized(self):
        rects = _tile_rects(2)
        for x1, y1, x2, y2 in rects:
            self.assertGreater(x2 - x1, 200)
            self.assertGreater(y2 - y1, 200)

    def test_tiles_shrink_inside_cells(self):
        # window rects must sit 8px inside their cells so frames never touch
        rects = _tile_rects(2)
        self.assertTrue(all(x1 >= 24 and y1 >= 24 for x1, y1, _, _ in rects))


class WindowRelockTests(unittest.TestCase):
    class _Rect:
        def __init__(self, left, top, right, bottom):
            self.left = left
            self.top = top
            self.right = right
            self.bottom = bottom

    class _Window:
        def __init__(self, rect):
            self._rect = rect
            self.moved = []

        def rectangle(self):
            return self._rect

    def test_relocks_when_window_drifted(self):
        window = self._Window(self._Rect(100, 100, 800, 900))  # drifted from tile
        with patch("photos_capture._fix_window") as fix:
            _ensure_window_rect(window, (120, 120, 760, 860))
        fix.assert_called_once()
        args = fix.call_args[0][1]
        self.assertEqual(tuple(args), (120, 120, 760, 860))

    def test_skips_when_window_in_place(self):
        window = self._Window(self._Rect(120, 120, 760, 860))
        with patch("photos_capture._fix_window") as fix:
            _ensure_window_rect(window, (120, 120, 760, 860))
        fix.assert_not_called()


class ParallelCaptureTests(unittest.TestCase):
    def test_parallel_captures_all_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            for page in ("1", "2", "3", "4"):
                _fake_image(images / f"{page}.jpg")

            def ok(*args, **kwargs):
                return _ok_manifest(), None

            with patch("photos_capture._capture_photos_fast_page", side_effect=ok):
                report = capture_photos_fast_batch(
                    images, out, pages={"1", "2", "3", "4"},
                    max_attempts=1, ask_fallback=False, parallel=2,
                )
            self.assertEqual(len(report["pages"]), 4)
            self.assertTrue(all(p["status"] == "captured" for p in report["pages"]))
            self.assertEqual(report["parallel"], 2)

    def test_frames_only_forwards_flag_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            _fake_image(images / "1.jpg")

            seen = {}

            def ok(*args, **kwargs):
                seen["frames_only"] = kwargs.get("frames_only")
                return _ok_manifest(frames_only=True), None

            with patch("photos_capture._capture_photos_fast_page", side_effect=ok):
                capture_photos_fast_batch(
                    images, out, pages={"1"},
                    max_attempts=1, ask_fallback=False, frames_only=True,
                )
            self.assertTrue(seen.get("frames_only"))
            manifest = (out / "1" / "capture.json").read_text(encoding="utf-8-sig")
            self.assertIn('"frames_only": true', manifest)

    def test_fixed_window_rect_passed_to_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            _fake_image(images / "1.jpg")

            seen = {}

            def ok(*args, **kwargs):
                seen["rect"] = kwargs.get("fixed_window_rect")
                return _ok_manifest(), None

            with patch("photos_capture._capture_photos_fast_page", side_effect=ok):
                capture_photos_fast_batch(
                    images, out, pages={"1"},
                    max_attempts=1, ask_fallback=False, fixed_window=True,
                )
            self.assertIsNotNone(seen.get("rect"))


if __name__ == "__main__":
    unittest.main()
