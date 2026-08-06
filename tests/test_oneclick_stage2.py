"""Stage 2 helpers: box joining, dropped false selections, typeset lines."""

import json
import tempfile
import unittest
from pathlib import Path

from one_click_translate import (
    blue_boxes_to_lines,
    build_page_texts,
    build_union_overrides,
)


def _island_dataset(islands):
    """Minimal dataset-like object exposing ``islands`` (objects with
    ``id`` / ``image_bbox`` attributes)."""
    class _Island:
        def __init__(self, data):
            self.id = data["id"]
            self.image_bbox = tuple(data["image_bbox"])

    class _Dataset:
        def __init__(self, items):
            self.islands = [_Island(item) for item in items]

    return _Dataset(islands)


def _unit(islands, blocks, groups):
    return {
        "islands": islands,
        "blocks": blocks,
        "groups": groups,
    }


class BuildPageTextsTests(unittest.TestCase):
    def setUp(self):
        self.islands = {
            "island_001": {"id": "island_001", "image_bbox": [0, 0, 1000, 1000]},
        }
        self.box_a = {"id": "a", "image_bbox": [100, 100, 200, 900]}
        self.box_b = {"id": "b", "image_bbox": [300, 100, 400, 900]}
        self.box_h1 = {"id": "h1", "image_bbox": [100, 100, 900, 150]}
        self.box_h2 = {"id": "h2", "image_bbox": [100, 200, 900, 250]}

    def test_vertical_join_right_to_left(self):
        units = {
            "1": _unit(
                self.islands,
                {"a": self.box_a, "b": self.box_b},
                {"g1": {"id": "g1", "members": ["a", "b"]}},
            )
        }
        readings = {"1:box:a": "右列", "1:box:b": "左列"}
        texts, directions = build_page_texts(units, readings, {"1": set()})
        # vertical: rightmost column (larger x) first
        self.assertEqual(texts["1"]["island_001"], "左列右列")
        self.assertEqual(directions["1"]["island_001"], "vertical")

    def test_dropped_box_excluded_from_join(self):
        units = {
            "1": _unit(
                self.islands,
                {"a": self.box_a, "b": self.box_b},
                {"g1": {"id": "g1", "members": ["a", "b"]}},
            )
        }
        readings = {"1:box:a": "右列", "1:box:b": ""}
        texts, _ = build_page_texts(units, readings, {"1": {"b"}})
        self.assertEqual(texts["1"]["island_001"], "右列")

    def test_all_dropped_yields_empty_text(self):
        units = {
            "1": _unit(
                self.islands,
                {"a": self.box_a, "b": self.box_b},
                {"g1": {"id": "g1", "members": ["a", "b"]}},
            )
        }
        readings = {"1:box:a": "", "1:box:b": ""}
        texts, _ = build_page_texts(units, readings, {"1": {"a", "b"}})
        self.assertEqual(texts["1"]["island_001"], "")

    def test_horizontal_join_top_to_bottom(self):
        units = {
            "1": _unit(
                self.islands,
                {"h1": self.box_h1, "h2": self.box_h2},
                {"g1": {"id": "g1", "members": ["h1", "h2"]}},
            )
        }
        readings = {"1:box:h1": "第一行", "1:box:h2": "第二行"}
        texts, directions = build_page_texts(units, readings, {"1": set()})
        self.assertEqual(texts["1"]["island_001"], "第一行第二行")
        self.assertEqual(directions["1"]["island_001"], "horizontal")


class BlueBoxesToLinesTests(unittest.TestCase):
    def test_exclude_drops_boxes(self):
        with tempfile.TemporaryDirectory(prefix="oneclick_lines_") as temporary:
            path = Path(temporary) / "selection_blocks.json"
            path.write_text(
                json.dumps(
                    {
                        "selection_blocks": [
                            {"id": "a", "image_bbox": [0, 0, 10, 100], "page_scale": False},
                            {"id": "b", "image_bbox": [20, 0, 30, 100], "page_scale": False},
                            {"id": "c", "image_bbox": [0, 200, 500, 600], "page_scale": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            lines = blue_boxes_to_lines(path, pad=6)
            self.assertEqual(len(lines), 2)
            lines_excluded = blue_boxes_to_lines(path, pad=6, exclude={"b"})
            self.assertEqual(len(lines_excluded), 1)
            polygon = lines_excluded[0].polygon
            self.assertEqual(polygon[0], (0.0, 0.0))
            self.assertEqual(polygon[2], (16.0, 106.0))


class BuildUnionOverridesTests(unittest.TestCase):
    def test_glyph_limits_use_padded_box_dimensions(self):
        # Island 100x500; box A 40 wide, box B 60 wide, both 300 tall.
        dataset = _island_dataset(
            [
                {"id": "island_001", "image_bbox": [0, 0, 100, 500]},
            ]
        )
        boxes = [
            {"id": "a", "image_bbox": [10, 20, 50, 320]},
            {"id": "b", "image_bbox": [60, 30, 120, 330]},
        ]
        overrides = build_union_overrides(dataset, boxes, pad=6)
        self.assertIn("island_001", overrides)
        # Padded widths: A = 40+12=52, B = 60+12=72 (clamped to island
        # right edge: B x2 = 120 -> 126 clamped to 100 -> width 46? No:
        # island bbox [0,0,100,500], so x2 clamps to 100-0=100, width =
        # (100) - (60-6=54) = 46.
        self.assertEqual(overrides["island_001"]["glyph_width_limit"], 46.0)
        # Padded heights: A = 300+12=312, B = 300+12=312 (y within island).
        self.assertEqual(overrides["island_001"]["glyph_height_limit"], 312.0)
        # Union area = padded boxes' pixel count.  A spans x[4,56],
        # B spans x[54,100] -> 2px overlap band over y[24,326].
        self.assertEqual(overrides["island_001"]["area"], 29972.0)

    def test_boxes_outside_island_are_ignored(self):
        dataset = _island_dataset(
            [
                {"id": "island_001", "image_bbox": [0, 0, 100, 100]},
            ]
        )
        boxes = [
            {"id": "a", "image_bbox": [10, 10, 50, 50]},
            {"id": "b", "image_bbox": [500, 500, 600, 600]},
        ]
        overrides = build_union_overrides(dataset, boxes, pad=6)
        self.assertIn("island_001", overrides)
        # Only box a's padded rect contributes: [4,4]-[56,56] = 52x52.
        self.assertEqual(overrides["island_001"]["area"], 2704.0)


if __name__ == "__main__":
    unittest.main()
