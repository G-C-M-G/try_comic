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
    """Active frame plus a selected frame with blue overlay rectangles.

    The selected frame applies the measured blend (keep*active + mask*alpha)
    inside every box, exactly as Photos renders the overlay.
    """
    active = np.full((size[0], size[1], 3), 128, dtype=np.uint8)
    selected = active.copy().astype(np.float32)
    for x, y, w, h in boxes:
        blended = _blend(128)
        selected[y:y + h, x:x + w] = blended
    return active, selected.astype(np.uint8)

class SelectionBoxTests(unittest.TestCase):
    def test_detects_each_atomic_box_separately(self):
        active, selected = _frames_with_blue_boxes(
            [(50, 30, 22, 90), (77, 31, 22, 90), (104, 30, 22, 90)]
        )
        blocks = detect_selection_boxes(active, selected)

        self.assertEqual(len(blocks), 3)
        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50, 77, 104])

    def test_splits_touching_components_at_projection_gap(self):
        # two boxes that touch but keep a one-column hole between them
        active, selected = _frames_with_blue_boxes(
            [(50, 30, 22, 90), (73, 31, 22, 90)]
        )
        selected[:, 72] = 128  # punch a hole (back to original) so the
        # boxes stay separate
        blocks = detect_selection_boxes(active, selected)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(sorted(b.x for b in blocks), [50, 73])

    def test_rejects_unchanged_pixels(self):
        # Pixels that did NOT change between the two frames are not part of
        # the overlay, whatever their colour.
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected[60:80, 150:180] = (20, 120, 240)  # blue but unchanged
        active[60:80, 150:180] = (20, 120, 240)
        blocks = detect_selection_boxes(active, selected)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50])

    def test_detects_non_blue_change(self):
        # ANY single-channel change over the threshold marks the pixel, even
        # when the changed colour is not blue (yellow-green here).
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected[60:80, 150:180, 2] = 200
        selected[60:80, 150:180, 0] = 20
        selected[60:80, 150:180, 1] = 240
        blocks = detect_selection_boxes(active, selected)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50, 150])

    def test_detects_changed_pixels(self):
        # Pixels that changed between the two frames are detected.
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected[30:120, 50:72] = _blend(128)
        blocks = detect_selection_boxes(active, selected)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50])

    def test_detects_dark_changed_pixels(self):
        # A dark changed pixel still counts: no brightness gate.
        dark = _blend(40)
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected[60:80, 150:180] = dark
        blocks = detect_selection_boxes(active, selected)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50, 150])

    def test_island_blocks_do_not_filter(self):
        # island_blocks is accepted but does not restrict the detection.
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected[60:80, 150:180] = _blend(128)
        islands = [OverlayBlock(40, 20, 60, 120)]  # covers only the first
        blocks = detect_selection_boxes(active, selected, island_blocks=islands)

        xs = sorted(block.x for block in blocks)
        self.assertEqual(xs, [50, 150])

    def test_drops_whole_page_selection(self):
        # A selection covering the whole frame (>20% area) is not an atomic
        # text box: dropped entirely.
        active, selected = _frames_with_blue_boxes([])
        selected[:, :] = _blend(128)
        blocks = detect_selection_boxes(active, selected)

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
        """base = patterned original; active = dimmed scan-text frame;
        selected = active everywhere EXCEPT the box rects, which get the
        exact blue overlay applied over the ORIGINAL pixels (as measured
        in real captures: outside the boxes both frames stay dimmed)."""
        rng = np.random.default_rng(7)
        base = rng.integers(60, 230, size=(size[0], size[1], 3), dtype=np.uint8)
        active = (base.astype(np.float64) * _KEEP).astype(np.uint8)
        selected = active.astype(np.float64)
        for x, y, w, h in boxes:
            selected[y:y + h, x:x + w] = blue_overlay_image(base[y:y + h, x:x + w])
        return base, active, selected.astype(np.uint8)

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
        selected[30:120, 50:72] = blue_overlay_image(base[30:120, 50:72])
        blocks = detect_selection_boxes(active, selected, base=base)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].x, 50)

    def test_formula_equals_delta_without_base(self):
        base, active, selected = self._frames([(50, 30, 22, 90)])
        with_base = detect_selection_boxes(active, selected, base=base)
        without_base = detect_selection_boxes(active, selected)
        self.assertEqual(len(with_base), 1)
        self.assertEqual(len(without_base), 1)
        self.assertEqual(with_base[0].x, without_base[0].x)


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
        active, selected = _frames_with_blue_boxes([(50, 30, 22, 90)])
        selected = np.pad(selected, ((0, 1), (0, 0), (0, 0)), mode="constant")
        blocks = detect_selection_boxes(active, selected)
        self.assertTrue(blocks)


if __name__ == "__main__":
    unittest.main()
