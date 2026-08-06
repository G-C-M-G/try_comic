import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from island_extract import (
    _cluster_by_y_layers,
    _selection_in_island,
    extract_islands,
)
from photos_capture import OverlayBlock


def _make_page(tmp, islands, dark_panels=(), glyphs=(), dims=(600, 800)):
    """Synthetic capture.

    ``islands``: highlight boxes (original pixels kept, no diff).
    ``dark_panels``: dark panel interiors inside boxes (kept at 60 both
    frames -> undimmed, but fail the old brightness guard).
    ``glyphs``: dark glyph rects inside islands (30 both frames).  Glyphs
    outside islands are drawn dark in base and gray in active (dimmed).
    """
    base = np.full((dims[0], dims[1], 3), 220, dtype=np.uint8)
    active = np.full((dims[0], dims[1], 3), 128, dtype=np.uint8)
    for x1, y1, x2, y2 in dark_panels:
        base[y1:y2, x1:x2] = 60
        active[y1:y2, x1:x2] = 60
    for x1, y1, x2, y2 in glyphs:
        base[y1:y2, x1:x2] = 30
        active[y1:y2, x1:x2] = 30
    for x1, y1, x2, y2 in islands:
        active[y1:y2, x1:x2] = base[y1:y2, x1:x2]
    Image.fromarray(base).save(tmp / "photos_base.png")
    Image.fromarray(active).save(tmp / "photos_ocr.png")


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


class DarkPanelTests(unittest.TestCase):
    def test_dark_panel_interior_belongs_to_island(self):
        root = _tmp()
        _make_page(root, [(200, 150, 260, 320)], dark_panels=[(200, 150, 260, 320)])
        base = np.array(Image.open(root / "photos_base.png").convert("RGB"))
        active = np.array(Image.open(root / "photos_ocr.png").convert("RGB"))
        # white box outline seeds the island (Photos draws it on dark panels)
        base[150:153, 200:260] = 230
        base[317:320, 200:260] = 230
        base[150:320, 200:203] = 230
        base[150:320, 257:260] = 230
        active[150:153, 200:260] = 230
        active[317:320, 200:260] = 230
        active[150:320, 200:203] = 230
        active[150:320, 257:260] = 230
        Image.fromarray(base).save(root / "photos_base.png")
        Image.fromarray(active).save(root / "photos_ocr.png")
        timing = {}
        result = extract_islands(root, (985, 1400), timing, min_island_area=0)
        self.assertEqual(len(result["islands"]), 1)
        bbox = result["islands"][0]["display_bbox"]
        self.assertLessEqual(bbox[1], 150)
        self.assertGreaterEqual(bbox[3], 320)
        self.assertTrue(result["island_mask"][200, 230])

    def test_dark_speck_outside_islands_is_not_an_island(self):
        root = _tmp()
        _make_page(root, [(200, 150, 260, 320)])
        base = np.array(Image.open(root / "photos_base.png").convert("RGB"))
        active = np.array(Image.open(root / "photos_ocr.png").convert("RGB"))
        # large undimmed dark patch far from any box: must not become an island
        base[500:560, 500:580] = 40
        active[500:560, 500:580] = 40
        Image.fromarray(base).save(root / "photos_base.png")
        Image.fromarray(active).save(root / "photos_ocr.png")
        timing = {}
        result = extract_islands(root, (985, 1400), timing, min_island_area=0)
        self.assertEqual(len(result["islands"]), 1)


class SelectionLayerSplitTests(unittest.TestCase):
    def test_clusters_boxes_by_y_layers(self):
        layers = _cluster_by_y_layers(
            [(100, 200, 130, 300), (140, 205, 170, 295), (100, 380, 130, 480)]
        )
        self.assertEqual(len(layers), 2)
        self.assertEqual(len(layers[0]), 2)
        self.assertEqual(len(layers[1]), 1)

    def test_selection_membership_by_coverage(self):
        island = (100, 150, 300, 500)
        inside = (160, 200, 190, 300)
        outside = (400, 400, 430, 500)
        self.assertTrue(_selection_in_island(island, inside))
        self.assertFalse(_selection_in_island(island, outside))

    def test_splits_island_on_y_layers(self):
        # Two vertically separate blue boxes inside one diff island become
        # two group islands; the diff island is mostly covered -> removed.
        root = _tmp()
        _make_page(root, [(150, 200, 250, 480)], glyphs=[
            (170, 230, 190, 300),
            (170, 410, 190, 470),
        ])
        timing = {}
        blocks = [
            OverlayBlock(168, 226, 24, 86),
            OverlayBlock(168, 406, 24, 76),
        ]
        result = extract_islands(
            root, (985, 1400), timing, min_island_area=0, selection_blocks=blocks
        )
        self.assertEqual(len(result["islands"]), 2)
        centers = sorted(
            (island["display_bbox"][1] + island["display_bbox"][3]) // 2
            for island in result["islands"]
        )
        self.assertGreater(centers[1] - centers[0], 150)

    def test_keeps_single_layer_island_unsplit(self):
        # Two same-layer boxes aggregate into ONE group island; the diff
        # island is mostly covered by the group -> replaced by the group.
        root = _tmp()
        _make_page(root, [(150, 200, 250, 300)], glyphs=[
            (170, 230, 190, 290),
            (200, 233, 220, 290),
        ])
        timing = {}
        blocks = [
            OverlayBlock(168, 228, 24, 64),
            OverlayBlock(198, 231, 24, 61),
        ]
        result = extract_islands(
            root, (985, 1400), timing, min_island_area=0, selection_blocks=blocks
        )
        self.assertEqual(len(result["islands"]), 1)
        self.assertTrue(result["islands"][0].get("from_selection_group"))


if __name__ == "__main__":
    unittest.main()
