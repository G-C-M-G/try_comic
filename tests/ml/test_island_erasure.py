from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_erasure import erase_island_image
from manga_localizer.eraser import ErasureConfig


def _dataset(islands: list[dict], size=(200, 300)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


def _white_page_with_black_text(size=(200, 300)):
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(15, 55, 5):
        draw.line((x, 30, x, 80), fill=(0, 0, 0), width=2)
    return image


class IslandErasureTest(unittest.TestCase):
    def test_polygon_island_erase_keeps_outside_untouched(self):
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
        image = _white_page_with_black_text()
        result = erase_island_image(image, dataset, config=ErasureConfig())
        # hard pixel-integrity guarantee
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertTrue(result.pixels.outside_mask_unchanged)
        self.assertEqual(result.island_pixels.changed_outside_mask_pixels, 0)
        # glyph pixels inside the island were erased
        self.assertGreater(result.island_pixels.changed_inside_mask_pixels, 0)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(result.columns[0].column_id, "island_001")
        self.assertEqual(result.columns[0].strategy, "solid_fill")
        # erased output differs from source only inside the island
        difference = result.difference
        self.assertEqual(difference.getpixel((5, 5)), (0, 0, 0))
        self.assertNotEqual(difference.getpixel((20, 40)), (0, 0, 0))

    def test_bbox_fallback_island(self):
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
        image = _white_page_with_black_text()
        result = erase_island_image(image, dataset, config=ErasureConfig())
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertEqual(result.columns[0].column_id, "island_002")
        # island has no ink, so nothing inside changes either
        self.assertEqual(result.island_pixels.changed_inside_mask_pixels, 0)

    def test_forced_inpaint_strategy(self):
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
        image = _white_page_with_black_text()
        result = erase_island_image(
            image, dataset, config=ErasureConfig(style_override="inpaint")
        )
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertEqual(result.columns[0].strategy, "inpaint")
        self.assertEqual(result.columns[0].classifier, "cli-override")


if __name__ == "__main__":
    unittest.main()
