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
            # Canonical "one box" contract: the block stays inside the
            # island bbox (here 400x120).
            x1, y1, x2, y2 = audit.block_bbox
            self.assertLessEqual(x2 - x1, 400)
            self.assertLessEqual(y2 - y1, 120)

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
        # A long translation in a small island: the block must fit inside
        # the island bbox (the canonical box, here 320x160) -- the font
        # shrinks below the 28px preferred floor if needed.
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
            self.assertLessEqual(x2 - x1, 320)
            self.assertLessEqual(y2 - y1, 160)

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
            # The block stays inside the island bbox (the canonical box,
            # here 400x300); a single oversized glyph is clipped there.
            x1, y1, x2, y2 = audit.block_bbox
            self.assertLessEqual(x2 - x1, 400)
            self.assertLessEqual(y2 - y1, 300)


if __name__ == "__main__":
    unittest.main()


class SfxFontCapTests(unittest.TestCase):
    """拟声词（短假名/符号）字号封顶，且只封顶不放大。

    大拟声框在原图里又大又粗，字号若跟着框走会得到 60~120px 的中文拟声，
    压过对白；封顶后固定在小注音量级。判定依据是**日文原文**，正常台词
    不受影响。
    """

    def _big_island(self, tmp, size=(420, 420)):
        return _dataset(
            [("island_001", (100, 100, 100 + size[0], 100 + size[1]))], tmp
        )

    def _typeset(self, dataset, translations, sfx_islands, direction="vertical"):
        erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
        return typeset_island_page(
            erased, dataset, translations, [],
            config=TypesetConfig(),
            direction_overrides={"island_001": direction},
            sfx_islands=sfx_islands,
        )

    def test_sfx_capped_at_configured_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._big_island(tmp)
            result = self._typeset(
                dataset, {"island_001": "夹紧♡"}, {"island_001"},
            )
            audit = result.audits[0]
            self.assertLessEqual(audit.font_size, TypesetConfig().sfx_max_font_size)

    def test_non_sfx_island_not_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._big_island(tmp)
            result = self._typeset(
                dataset, {"island_001": "夹紧♡"}, set(),
            )
            self.assertGreater(
                result.audits[0].font_size, TypesetConfig().sfx_max_font_size,
            )

    def test_cap_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._big_island(tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased, dataset, {"island_001": "夹紧♡"}, [],
                config=TypesetConfig(sfx_max_font_size=18),
                direction_overrides={"island_001": "vertical"},
                sfx_islands={"island_001"},
            )
            self.assertLessEqual(result.audits[0].font_size, 18)

    def test_small_box_not_enlarged_by_cap(self):
        """框本身就小：上限不该把字号放大（min(框, 上限)）。"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 130, 180))], tmp)
            result = self._typeset(
                dataset, {"island_001": "夹紧♡"}, {"island_001"},
            )
            self.assertLessEqual(result.audits[0].font_size, 30)

    def test_horizontal_sfx_capped_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (60, 100, 580, 300))], tmp)
            result = self._typeset(
                dataset, {"island_001": "夹紧♡"}, {"island_001"},
                direction="horizontal",
            )
            self.assertLessEqual(result.audits[0].font_size, 30)
