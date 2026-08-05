from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from manga_localizer.island_contract import dataset_from_data, filter_tiny_islands
from manga_localizer.island_erasure import erase_island_image
from manga_localizer.eraser import ErasureConfig


def _dataset(islands, size=(200, 300)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


class FilterTinyIslandsTest(unittest.TestCase):
    def test_drops_below_threshold(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 60, 60],
                    "image_center": [35, 35],
                    "image_area": 2500.0,
                },
                {
                    "id": "island_002",
                    "image_bbox": [100, 10, 130, 40],
                    "image_center": [115, 25],
                    "image_area": 900.0,
                },
                {
                    "id": "island_003",
                    "image_bbox": [150, 10, 158, 34],
                    "image_center": [154, 22],
                    "image_area": 132.0,
                },
            ]
        )
        filtered, dropped = filter_tiny_islands(dataset, 800.0)
        self.assertEqual([island.id for island in filtered.islands], ["island_001", "island_002"])
        self.assertEqual([island.id for island in dropped], ["island_003"])
        self.assertNotEqual(filtered.effective_sha256, dataset.effective_sha256)

    def test_no_threshold_keeps_all(self):
        dataset = _dataset(
            [
                {
                    "id": "island_003",
                    "image_bbox": [150, 10, 158, 34],
                    "image_center": [154, 22],
                    "image_area": 132.0,
                }
            ]
        )
        filtered, dropped = filter_tiny_islands(dataset, 0.0)
        self.assertIs(filtered, dataset)
        self.assertEqual(dropped, ())


class ErasureMinAreaTest(unittest.TestCase):
    def test_dropped_island_is_untouched(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 60, 60],
                    "image_center": [35, 35],
                    "image_area": 2500.0,
                    "image_polygon": [[10, 10], [60, 10], [60, 60], [10, 60]],
                },
                {
                    "id": "island_002",
                    "image_bbox": [150, 10, 158, 34],
                    "image_center": [154, 22],
                    "image_area": 132.0,
                },
            ]
        )
        image = Image.new("RGB", (200, 300), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.line((20, 20, 20, 50), fill=(0, 0, 0), width=2)
        draw.line((152, 12, 152, 32), fill=(0, 0, 0), width=2)
        result = erase_island_image(
            image, dataset, config=ErasureConfig(), min_island_area=800.0
        )
        self.assertEqual(result.dropped_tiny, ("island_002",))
        self.assertEqual(len(result.columns), 1)
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        # dropped island's ink stays exactly as it was
        self.assertEqual(result.image.getpixel((152, 20)), (0, 0, 0))
        self.assertEqual(result.image.getpixel((154, 20)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
