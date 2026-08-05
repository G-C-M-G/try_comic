from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image

from manga_localizer.errors import ContractValidationError, ImageDimensionError
from manga_localizer.island_contract import (
    dataset_from_data,
    expand_island_bboxes,
    load_island_dataset,
    validate_dataset_images,
)

UPSTREAM = Path(r"D:\workplace\opencode\try_comic")
REAL_DATASET = UPSTREAM / "outputs" / "exam" / "dataset33" / "33"


def _synthetic(islands: list[dict], size=(200, 300), **extra) -> dict:
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "fill_black": [0, 0, 0],
        "contours": True,
        "islands": islands,
    }
    data.update(extra)
    return data


class IslandContractSyntheticTest(unittest.TestCase):
    def test_parses_valid_islands(self):
        data = _synthetic(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 3500.0,
                    "image_polygon": [[10, 20], [60, 20], [60, 90], [10, 90]],
                },
                {
                    "id": "island_002",
                    "image_bbox": [100, 30, 150, 120],
                    "image_center": [125, 75],
                    "image_area": 4500.0,
                },
            ]
        )
        dataset = dataset_from_data(data, path=Path("x/islands.json"), source_sha256="a" * 64)
        self.assertEqual(dataset.image_size, (200, 300))
        self.assertEqual(len(dataset.islands), 2)
        self.assertEqual(dataset.islands[0].polygon[0], (10.0, 20.0))
        self.assertIsNone(dataset.islands[1].polygon)

    def test_rejects_empty_island_list(self):
        with self.assertRaises(ContractValidationError):
            dataset_from_data(_synthetic([]), path=Path("x"), source_sha256="a" * 64)

    def test_rejects_duplicate_ids(self):
        island = {
            "id": "island_001",
            "image_bbox": [10, 20, 60, 90],
            "image_center": [35, 55],
            "image_area": 1.0,
        }
        with self.assertRaises(ContractValidationError):
            dataset_from_data(
                _synthetic([island, dict(island)]), path=Path("x"), source_sha256="a" * 64
            )

    def test_rejects_bbox_outside_image(self):
        data = _synthetic(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 250, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                }
            ]
        )
        with self.assertRaises(ContractValidationError):
            dataset_from_data(data, path=Path("x"), source_sha256="a" * 64)

    def test_rejects_zero_area_polygon(self):
        data = _synthetic(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                    "image_polygon": [[10, 20], [60, 20], [10, 20]],
                }
            ]
        )
        with self.assertRaises(ContractValidationError):
            dataset_from_data(data, path=Path("x"), source_sha256="a" * 64)

    def test_rejects_bad_image_size(self):
        with self.assertRaises(ContractValidationError):
            dataset_from_data(
                _synthetic([], size=(0, 0)), path=Path("x"), source_sha256="a" * 64
            )

    def test_rejects_noninteger_bbox(self):
        data = _synthetic(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60.5, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                }
            ]
        )
        with self.assertRaises(ContractValidationError):
            dataset_from_data(data, path=Path("x"), source_sha256="a" * 64)


class IslandContractImageTest(unittest.TestCase):
    def test_expand_island_bboxes(self):
        data = _synthetic(
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
        dataset = dataset_from_data(data, path=Path("x"), source_sha256="a" * 64)
        expanded = expand_island_bboxes(
            dataset,
            expand_ids=["island_001"],
            pad=10,
            fragment_boxes={"island_001": [[5, 8, 70, 105]]},
        )
        self.assertEqual(expanded.islands_by_id["island_001"].image_bbox, (0, 6, 72, 107))
        self.assertEqual(expanded.islands_by_id["island_002"].image_bbox, (100, 30, 150, 120))
        self.assertNotEqual(expanded.effective_sha256, dataset.effective_sha256)
        # padding is clamped to the image
        expanded_edge = expand_island_bboxes(dataset, expand_ids=["island_002"], pad=500)
        self.assertEqual(
            expanded_edge.islands_by_id["island_002"].image_bbox, (0, 0, 200, 300)
        )

    def test_expand_noop_without_targets(self):
        data = _synthetic(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                }
            ]
        )
        dataset = dataset_from_data(data, path=Path("x"), source_sha256="a" * 64)
        self.assertIs(expand_island_bboxes(dataset, expand_ids=[], pad=10), dataset)

    def test_image_size_mismatch_rejected(self):
        dataset_dir = Path(__file__).resolve().parent / "tmp_island_images"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
            (dataset_dir / "islands.json").write_text(
                json.dumps(
                    _synthetic(
                        [
                            {
                                "id": "island_001",
                                "image_bbox": [10, 20, 60, 90],
                                "image_center": [35, 55],
                                "image_area": 1.0,
                            }
                        ],
                        size=(200, 300),
                    )
                ),
                encoding="utf-8",
            )
            Image.new("RGB", (200, 300), (255, 255, 255)).save(
                dataset_dir / "filled_black.png"
            )
            Image.new("RGB", (100, 100), (0, 0, 0)).save(dataset_dir / "filled_blue.png")
            dataset = load_island_dataset(dataset_dir / "islands.json")
            with self.assertRaises(ImageDimensionError):
                validate_dataset_images(dataset, dataset_dir)
        finally:
            for name in ("islands.json", "filled_black.png", "filled_blue.png"):
                (dataset_dir / name).unlink(missing_ok=True)
            dataset_dir.rmdir()

    def test_missing_file_rejected(self):
        with self.assertRaises(ContractValidationError):
            load_island_dataset(Path(__file__).resolve().parent / "does_not_exist.json")


@unittest.skipUnless(REAL_DATASET.is_dir(), "upstream dataset33 not present")
class IslandContractRealTest(unittest.TestCase):
    def test_loads_dataset33(self):
        dataset = load_island_dataset(REAL_DATASET / "islands.json")
        self.assertEqual(dataset.image_size, (985, 1400))
        self.assertEqual(len(dataset.islands), 17)
        ids = {island.id for island in dataset.islands}
        self.assertEqual(len(ids), 17)
        with_polygon = [island for island in dataset.islands if island.polygon]
        self.assertGreater(len(with_polygon), 0)

    def test_images_match_dataset_size(self):
        dataset = load_island_dataset(REAL_DATASET / "islands.json")
        images = validate_dataset_images(dataset, REAL_DATASET)
        self.assertEqual(images["filled_black.png"].name, "filled_black.png")
        self.assertEqual(images["original.jpg"].name, "original.jpg")


if __name__ == "__main__":
    unittest.main()
