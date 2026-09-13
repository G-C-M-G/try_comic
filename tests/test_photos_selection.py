import unittest

import numpy as np
from photos_capture import (
    OverlayBlock,
    _align_frames,
    aggregate_selection_boxes,
    blue_overlay_image,
    detect_overlay_blocks,
    detect_selection_boxes,
)

_KEEP = 174.0 / 255.0
_ALPHA = 81.0 / 255.0
_MASK = (0.0, 120.0, 211.0)


def _blend(pixel, keep=_KEEP, alpha=_ALPHA, mask=_MASK):
    """Apply the measured overlay blend to a pixel (returns uint8 tuple)."""
    if isinstance(pixel, (int, np.integer)):
        pixel = (int(pixel),) * 3
    out = tuple(pixel[c] * keep + mask[c] * alpha for c in range(3))
    return tuple(int(round(v)) for v in out)


def _frames_with_blue_boxes(boxes, size=(300, 400)):
    """base/active/selected triplet with the measured overlay applied
    inside every box (as Photos renders it over the SCAN-TEXT view).

    * base = the original (random-ish) image;
    * active = the dimmed scan-text frame: active = base * keep everywhere
      (Photos 2026 dims the non-text areas of the scan-text view);
    * selected = active everywhere EXCEPT the box rects, which get the
      exact blue overlay applied over the ACTIVE (dimmed) pixels.
    """
    rng = np.random.default_rng(7)
    base = rng.integers(60, 230, size=(size[0], size[1], 3), dtype=np.uint8)
    active = (base.astype(np.float64) * _KEEP).astype(np.uint8)
    selected = active.astype(np.float64)
    for x, y, w, h in boxes:
        selected[y:y + h, x:x + w] = blue_overlay_image(active[y:y + h, x:x + w])
    return base, active, selected.astype(np.uint8)


class SelectionBoxTests(unittest.TestCase):
    def test_detects_each_atomic_box_separately(self):
        base, active, selected = _frames_with_blue_boxes(
            [(50, 30, 22, 90), (77, 31, 22, 90), (104, 30, 22, 90)]
        )
        blocks = detect_selection_boxes(active, selected, base=base)

        self.assertEqual(len(blocks), 3)
        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50, 77, 104])

    def test_splits_touching_components_at_projection_gap(self):
        # two boxes that touch but keep a one-column hole between them
        base, active, selected = _frames_with_blue_boxes(
            [(50, 30, 22, 90), (73, 31, 22, 90)]
        )
        selected[:, 72] = selected[0, 0, 0]  # punch a hole at the seam
        blocks = detect_selection_boxes(active, selected, base=base)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(sorted(b.x for b in blocks), [50, 73])

    def test_rejects_unchanged_pixels(self):
        # Pixels that did NOT change between base and selected are not
        # part of the overlay: the paint-delta guard drops them.
        base, active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        # Paint a blue region on the ORIGINAL too -- the overlay formula
        # would match it pixel-perfectly, but it is NOT repainted.
        base[60:80, 150:180] = (20, 120, 240)
        active[60:80, 150:180] = (
            np.array((20, 120, 240), dtype=np.float64) * _KEEP
        ).astype(np.uint8)
        selected[60:80, 150:180] = active[60:80, 150:180]
        blocks = detect_selection_boxes(active, selected, base=base)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50])

    def test_returns_empty_without_base(self):
        # Single-path detector has no delta fallback: without the
        # ORIGINAL-pixel baseline we cannot reconstruct the expected
        # overlay, so a base-less call returns an empty list and the
        # page is gated to the review queue.
        base, active, selected = _frames_with_blue_boxes(
            [(50, 30, 22, 90), (200, 100, 25, 35)]
        )
        # The detector must not invent boxes from frame deltas alone.
        self.assertEqual(detect_selection_boxes(active, selected), [])
        # Same result when both frames carry noise.
        noisy_active = (active.astype(np.float32) * 0.9).astype(np.uint8)
        self.assertEqual(detect_selection_boxes(noisy_active, selected), [])

    def test_drops_whole_page_selection(self):
        # A selection covering the whole frame (>20% area) is not an atomic
        # text box: dropped entirely.
        base, active, selected = _frames_with_blue_boxes([])
        # Make the entire selected frame a big overlay rectangle.
        selected = (active.astype(np.float64)).copy()
        selected[:, :] = blue_overlay_image(active)
        blocks = detect_selection_boxes(active, selected.astype(np.uint8), base=base)
        self.assertEqual(blocks, [])

    def test_aggregates_horizontal_neighbours_into_one_group(self):
        boxes = [
            OverlayBlock(50, 30, 22, 90),
            OverlayBlock(77, 31, 22, 90),
            OverlayBlock(104, 30, 22, 90),
        ]
        groups = aggregate_selection_boxes(boxes)

        self.assertEqual(len(groups), 1)
        bbox, members = groups[0]
        self.assertEqual(bbox, (50, 30, 126, 121))
        self.assertEqual(len(members), 3)

    def test_keeps_vertically_split_halves_in_separate_groups(self):
        boxes = [
            OverlayBlock(50, 30, 22, 90),
            OverlayBlock(51, 145, 22, 90),  # different text, 25px y gap
        ]
        groups = aggregate_selection_boxes(boxes)

        self.assertEqual(len(groups), 2)

    def test_keeps_far_boxes_in_separate_groups(self):
        boxes = [
            OverlayBlock(50, 30, 22, 90),
            OverlayBlock(200, 31, 22, 90),
        ]
        groups = aggregate_selection_boxes(boxes)

        self.assertEqual(len(groups), 2)

    def test_keeps_distant_rows_in_separate_groups(self):
        boxes = [
            OverlayBlock(50, 30, 22, 90),
            OverlayBlock(51, 400, 22, 90),
        ]
        groups = aggregate_selection_boxes(boxes, y_pad=8)

        self.assertEqual(len(groups), 2)


class BlueOverlayFormulaTests(unittest.TestCase):
    def _frames(self, boxes, size=(300, 400)):
        """base/active/selected as in ``_frames_with_blue_boxes``."""
        return _frames_with_blue_boxes(boxes, size)

    def test_formula_detects_boxes_with_base(self):
        base, active, selected = self._frames([(50, 30, 22, 90), (104, 30, 22, 90)])
        blocks = detect_selection_boxes(active, selected, base=base)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(sorted(b.x for b in blocks), [50, 104])

    def test_formula_ignores_untouched_blue_artwork(self):
        # A pure-blue region that was NOT repainted stays dimmed in the
        # selected frame: the overlay formula must not turn it into a box.
        blue = (0, 120, 211)
        base = np.full((400, 300, 3), 200, dtype=np.uint8)
        base[100:140, 150:210] = blue
        active = (base.astype(np.float64) * _KEEP).astype(np.uint8)
        selected = active.astype(np.float64)
        selected[30:120, 50:72] = blue_overlay_image(active[30:120, 50:72])
        blocks = detect_selection_boxes(active, selected, base=base)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].x, 50)

    def test_formula_returns_empty_without_base(self):
        # No base = no strict-formula detection (no delta fallback).
        base, active, selected = self._frames([(50, 30, 22, 90)])
        self.assertEqual(detect_selection_boxes(active, selected), [])


class FrameAlignTests(unittest.TestCase):
    def test_aligns_tiny_layout_drift(self):
        first = np.zeros((100, 120, 3), dtype=np.uint8)
        second = np.zeros((101, 120, 3), dtype=np.uint8)
        a, b = _align_frames(first, second)
        self.assertEqual(a.shape, (100, 120, 3))
        self.assertEqual(b.shape, (100, 120, 3))

    def test_rejects_large_mismatch(self):
        first = np.zeros((100, 120, 3), dtype=np.uint8)
        second = np.zeros((200, 120, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            _align_frames(first, second)

    def test_overlay_blocks_tolerate_1px_drift(self):
        base = np.full((120, 160, 3), 220, dtype=np.uint8)
        active = np.full((121, 160, 3), 128, dtype=np.uint8)
        active[:120, 10:55, :] = base[:120, 10:55, :]
        blocks = detect_overlay_blocks(base, active)
        self.assertTrue(blocks)

    def test_selection_boxes_tolerate_1px_drift(self):
        base, active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected = np.pad(selected, ((0, 1), (0, 0), (0, 0)), mode="constant")
        blocks = detect_selection_boxes(active, selected, base=base)
        self.assertTrue(blocks)


if __name__ == "__main__":
    unittest.main()
