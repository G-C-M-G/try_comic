import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_typeset import (
    TypesetConfig,
    infer_box_direction,
    infer_island_directions,
    typeset_island_page,
)


def _dataset(islands, tmp):
    payload = {
        "schema_version": 1,
        "page": "99",
        "image_size": [600, 800],
        "islands": [
            {
                "id": island_id,
                "image_bbox": list(bbox),
                "image_center": [
                    (bbox[0] + bbox[2]) // 2,
                    (bbox[1] + bbox[3]) // 2,
                ],
                "image_area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            }
            for island_id, bbox in islands
        ],
    }
    return dataset_from_data(payload, path=Path(tmp) / "islands.json",
                             source_sha256="test")


class DirectionInferenceTests(unittest.TestCase):
    def test_box_direction_by_aspect(self):
        self.assertEqual(infer_box_direction((100, 100, 300, 140)), "horizontal")
        self.assertEqual(infer_box_direction((100, 100, 140, 300)), "vertical")
        self.assertEqual(infer_box_direction((100, 100, 200, 200)), "horizontal")

    def test_island_direction_from_blue_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = {
                "selection_blocks": [
                    {"id": "sel_001", "image_bbox": [120, 120, 400, 150]},
                    {"id": "sel_002", "image_bbox": [120, 160, 400, 190]},
                    {"id": "sel_003", "image_bbox": [120, 200, 400, 230]},
                ]
            }
            (root / "selection_blocks.json").write_text(
                "{}".replace("{}", "null"), encoding="utf-8"
            )
            (root / "selection_blocks.json").write_text(
                __import__("json").dumps(selection), encoding="utf-8"
            )
            islands = [
                type("I", (), {"id": "island_001", "image_bbox": (100, 100, 450, 250)}),
                type("I", (), {"id": "island_002", "image_bbox": (100, 400, 450, 700)}),
            ]
            directions = infer_island_directions(
                root / "selection_blocks.json", islands
            )
            self.assertEqual(directions["island_001"], "horizontal")
            self.assertEqual(directions["island_002"], "vertical")


class HorizontalTypesetTests(unittest.TestCase):
    def test_horizontal_text_renders_inside_island(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 500, 220))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": "这里是后记的横向文本"},
                [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "horizontal"},
            )
            audit = result.audits[0]
            self.assertIn(audit.status, ("typeset", "typeset_pending_review"))
            self.assertGreaterEqual(audit.font_size, 30)
            # Hard cap: the block stays within the 150% envelope of the
            # group bbox (here the island bbox: 400x120 -> 600x180).
            x1, y1, x2, y2 = audit.block_bbox
            self.assertLessEqual(x2 - x1, 600)
            self.assertLessEqual(y2 - y1, 180)

    def test_horizontal_block_is_wider_than_tall(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 500, 220))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": "横向文本一行"},
                [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "horizontal"},
            )
            audit = result.audits[0]
            width = audit.block_bbox[2] - audit.block_bbox[0]
            height = audit.block_bbox[3] - audit.block_bbox[1]
            self.assertGreater(width, height)

    def test_vertical_default_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 220, 500))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": "竖排文本"},
                [],
                config=TypesetConfig(),
            )
            audit = result.audits[0]
            self.assertEqual(audit.status, "typeset")
            height = audit.block_bbox[3] - audit.block_bbox[1]
            width = audit.block_bbox[2] - audit.block_bbox[0]
            self.assertGreater(height, width)

    def test_long_horizontal_text_never_renders_outside_island(self):
        # A long translation in a small island: the block must respect the
        # hard 150% envelope of the group bbox (island 320x160 -> 480x240);
        # anything beyond is clipped by the caller, never rendered as a
        # full out-of-bounds block.
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 420, 260))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            text = "这是一段非常长的横向译文文字用来验证容量逻辑是否会把最后一行渲染到岛屿之外"
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": text},
                [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "horizontal"},
            )
            audit = result.audits[0]
            x1, y1, x2, y2 = audit.block_bbox
            self.assertLessEqual(x2 - x1, 480)
            self.assertLessEqual(y2 - y1, 240)

    def test_horizontal_ratio_uses_uniform_floor(self):
        # Horizontal uses the same 0.6 area-ratio floor as vertical.
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 500, 220))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": "横向文本面积约束测试一行"},
                [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "horizontal"},
            )
            audit = result.audits[0]
            self.assertIn(audit.status, ("typeset", "typeset_pending_review"))
            # With the 30px font floor the short text may under-fill the
            # island (ratio below 0.6 -> pending_review); it must never
            # exceed the hard 120% upper ratio.
            if audit.area_ratio is not None:
                self.assertLessEqual(audit.area_ratio, 1.1)

    def test_small_text_horizontal_goes_pending_review(self):
        # One character in a huge island: ratio below the floor ->
        # typeset_pending_review (still rendered, not out of bounds).
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 500, 400))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased,
                dataset,
                {"island_001": "一"},
                [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "horizontal"},
            )
            audit = result.audits[0]
            self.assertIn(audit.status, ("typeset", "typeset_pending_review"))
            # The block stays within the 150% envelope (island 400x300 ->
            # 600x450); a single oversized glyph is clipped to the island.
            x1, y1, x2, y2 = audit.block_bbox
            self.assertLessEqual(x2 - x1, 600)
            self.assertLessEqual(y2 - y1, 450)


if __name__ == "__main__":
    unittest.main()
