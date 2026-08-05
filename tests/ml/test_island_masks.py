from __future__ import annotations

import unittest

from PIL import Image

from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_masks import build_island_masks, island_mask_image


def _dataset(islands: list[dict], size=(200, 300)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


class IslandMasksTest(unittest.TestCase):
    def test_polygon_mask_pixels(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                    "image_polygon": [[10, 20], [60, 20], [60, 90], [10, 90]],
                }
            ]
        )
        mask = island_mask_image(dataset.islands[0], (200, 300))
        self.assertEqual(mask.size, (200, 300))
        # PIL fills inclusive edges: x 10..60 and y 20..90
        self.assertEqual(mask.histogram()[255], 51 * 71)
        self.assertEqual(mask.getpixel((11, 21)), 255)
        self.assertEqual(mask.getpixel((5, 5)), 0)

    def test_bbox_fallback_mask(self):
        dataset = _dataset(
            [
                {
                    "id": "island_002",
                    "image_bbox": [100, 30, 150, 120],
                    "image_center": [125, 75],
                    "image_area": 1.0,
                }
            ]
        )
        mask = island_mask_image(dataset.islands[0], (200, 300))
        self.assertEqual(mask.histogram()[255], 51 * 91)

    def test_build_island_masks_union(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                },
                {
                    "id": "island_002",
                    "image_bbox": [100, 30, 150, 120],
                    "image_center": [125, 75],
                    "image_area": 1.0,
                },
            ]
        )
        masks = build_island_masks(dataset)
        self.assertEqual(len(masks.columns), 2)
        self.assertEqual(masks.union_pixel_count, 51 * 71 + 51 * 91)
        self.assertEqual(masks.audits[0].fill_mode, "bbox")
        self.assertEqual(masks.audits[0].island_id, "island_001")
        self.assertEqual(masks.columns[0].audit.unit_id, "island_001")
        # union mask must contain both islands
        self.assertEqual(masks.combined.getpixel((11, 21)), 255)
        self.assertEqual(masks.combined.getpixel((101, 31)), 255)
        self.assertEqual(masks.combined.getpixel((5, 5)), 0)


if __name__ == "__main__":
    unittest.main()
