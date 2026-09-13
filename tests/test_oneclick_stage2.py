"""Stage 2 helpers: box joining, dropped false selections, typeset lines."""

import json
import tempfile
import unittest
from pathlib import Path

from one_click_translate import (
    _group_boxes_by_overlap,
    _load_readings_cache,
    _select_crop_misses,
    _store_readings_cache,
    blue_boxes_to_lines,
    build_page_texts,
    build_union_overrides,
)


def _box(box_id, x1, y1, x2, y2):
    return {"id": box_id, "image_bbox": [x1, y1, x2, y2], "page_scale": False}


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

    def test_bbox_is_unclamped_pad6_union_in_image_coords(self):
        # The canonical "one box": the union of the island's blue boxes
        # expanded 6 px per side, in IMAGE coordinates, NOT clamped to the
        # island bbox (the erasure/layout range may extend past the raw
        # islands.json bbox).
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
        # union = [10,20]-[120,330] expanded by 6 each side
        self.assertEqual(
            overrides["island_001"]["bbox"], (4, 14, 126, 336)
        )


class OcrReadingsCacheTests(unittest.TestCase):
    def _cache_path(self, tmp):
        return Path(tmp) / "ocr_readings.json"

    def test_store_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store_readings_cache(root, {"1:box:a": "右列", "1:box:b": "左列"})
            self.assertEqual(
                _load_readings_cache(root), {"1:box:a": "右列", "1:box:b": "左列"}
            )

    def test_empty_readings_never_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store_readings_cache(root, {"1:box:a": "", "1:box:b": "  "})
            self.assertEqual(_load_readings_cache(root), {})

    def test_store_merges_with_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store_readings_cache(root, {"1:box:a": "右列"})
            _store_readings_cache(root, {"1:box:b": "左列"})
            self.assertEqual(
                _load_readings_cache(root),
                {"1:box:a": "右列", "1:box:b": "左列"},
            )

    def test_corrupt_cache_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._cache_path(tmp).write_text("{not json", encoding="utf-8")
            self.assertEqual(_load_readings_cache(root), {})

    def test_missing_cache_file_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_load_readings_cache(Path(tmp)), {})

    def test_select_misses_only_uncached_non_empty(self):
        cached = {"1:box:a": "右列", "1:box:b": ""}
        crops = [
            {"id": "1:box:a", "source": "x", "bbox": [0, 0, 1, 1]},
            {"id": "1:box:b", "source": "x", "bbox": [0, 0, 1, 1]},
            {"id": "1:box:c", "source": "x", "bbox": [0, 0, 1, 1]},
        ]
        misses = _select_crop_misses(crops, cached)
        self.assertEqual([crop["id"] for crop in misses], ["1:box:b", "1:box:c"])


class GroupBoxesByOverlapTests(unittest.TestCase):
    def test_vertical_side_by_side_aligned_boxes_merge(self):
        # Two vertical boxes: padded gap negative (the 6px expansions
        # overlap) AND padded overlap > 40% of the LARGER padded height
        # -> one OCR unit.
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 80, 60, 110, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_boxes_within_12px_raw_gap_merge(self):
        # Page-12 style: narrow columns 9px apart -- the 6px expansions
        # overlap (negative padded gap), so the gap axis passes and the
        # overlap decides.
        boxes = [
            _box("a", 1084, 64, 1104, 220),
            _box("b", 1113, 64, 1136, 174),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_gap_within_60_percent_raw_width_merges(self):
        # 合并判据在**原始框**上：间距 15px < 60% × 较小框宽 (30px → 18px)。
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 95, 40, 125, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_gap_beyond_60_percent_raw_width_does_not_merge(self):
        # 同一对框把间距拉到 20px：20 > 60% × 30px = 18px -> 不并。
        # （旧规则在"扩边后"几何上判，扩边给间距加了 2×pad 的常量，
        #  等价比例约 20%；改成原始框后重标定为 60%。）
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 100, 40, 130, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_wide_horizontal_gap_does_not_merge(self):
        # 间距 24px > 60% × 较小框宽 (36px → 21.6px) -> 不并。
        boxes = [
            _box("a", 40, 30, 100, 200),
            _box("b", 124, 40, 160, 210),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_overlap_measured_against_larger_padded_height(self):
        # A short box overlapping a tall box by more than 40% of the SHORT
        # box but less than 40% of the LARGER padded height -> no merge
        # (the overlap condition is a real filter, not always satisfied).
        boxes = [
            _box("a", 50, 30, 80, 400),   # tall
            _box("b", 80, 60, 110, 140),  # short, top-aligned with a
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_same_column_stacked_never_merge(self):
        # Vertical boxes stacked along the reading direction stay separate
        # (the padded y-overlap is 2px < 40% of the larger padded height).
        boxes = [
            _box("a", 50, 30, 80, 120),
            _box("b", 52, 130, 82, 200),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_low_height_overlap_does_not_merge(self):
        # Padded y-overlap 52px < 40% of the larger padded height (182px)
        # -> no merge.
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 80, 160, 106, 260),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_horizontal_stacked_aligned_boxes_merge(self):
        # Two horizontal boxes: padded vertical gap negative AND padded
        # horizontal overlap > 40% of the LARGER padded width -> one unit.
        boxes = [
            _box("a", 40, 100, 240, 140),
            _box("b", 60, 140, 260, 176),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_horizontal_side_by_side_never_merge(self):
        # Horizontal boxes on the same row (reading direction): the padded
        # x-overlap is 2px < 40% of the larger padded width -> separate.
        boxes = [
            _box("a", 40, 100, 150, 140),
            _box("b", 160, 105, 280, 145),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_horizontal_wide_vertical_gap_does_not_merge(self):
        # Raw vertical gap 35px -> padded gap 23px > 20% of the smaller
        # padded height (52px * 0.2 = 10.4) -> no merge.
        boxes = [
            _box("a", 40, 100, 240, 140),
            _box("b", 60, 175, 260, 215),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_mixed_orientation_pairs_never_merge(self):
        # A vertical box overlapping a horizontal box is never one group.
        boxes = [
            _box("a", 40, 40, 140, 300),   # vertical
            _box("b", 100, 130, 300, 180),  # horizontal, overlaps a on x
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)


if __name__ == "__main__":
    unittest.main()
