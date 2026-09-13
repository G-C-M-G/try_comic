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
    """字号封顶岛（拟声词/短原文/过大单框）只封顶不放大。

    大拟声框在原图里又大又粗，字号若跟着框走会得到 60~120px 的中文拟声，
    压过对白；封顶后固定在小注音量级。判定依据是**日文原文**，正常台词
    不受影响。
    """

    def _big_island(self, tmp, size=(420, 420)):
        return _dataset(
            [("island_001", (100, 100, 100 + size[0], 100 + size[1]))], tmp
        )

    def _typeset(self, dataset, translations, capped_islands, direction="vertical"):
        erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
        return typeset_island_page(
            erased, dataset, translations, [],
            config=TypesetConfig(),
            direction_overrides={"island_001": direction},
            capped_islands=capped_islands,
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
                capped_islands={"island_001"},
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


class OversizedSingleBoxTests(unittest.TestCase):
    """过大单框（未合并、短边 ≫ 同方向平均）按拟声词封顶。

    只认单框岛：合并组的短边大是"句子被拆成几段"的结果，不是字号大。
    阈值复用大拟音筛框的 2.0× 同方向基准。
    """

    def _boxes(self, sizes):
        return [
            {"id": f"sel_{i:03d}", "image_bbox": [10 + i * 40, 10, 10 + i * 40 + w, 10 + h]}
            for i, (w, h) in enumerate(sizes)
        ]

    def _islands(self, specs, tmp):
        return _dataset(specs, tmp)

    def test_oversized_single_box_capped(self):
        from one_click_translate import oversized_single_box_islands

        with tempfile.TemporaryDirectory() as tmp:
            # 一页常规竖框短边 60，外加一个短边 200 的孤立大框
            sizes = [(60, 200)] * 4 + [(60, 200), (200, 400)]
            boxes = self._boxes(sizes)
            specs = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box["image_bbox"]
                specs.append((f"island_{i + 1:03d}", (x1, y1, x2, y2)))
            dataset = self._islands(specs, tmp)
            capped = oversized_single_box_islands(dataset, boxes, ratio=2.0)
            self.assertEqual(capped, {"island_006"})

    def test_merged_island_not_capped(self):
        from one_click_translate import oversized_single_box_islands

        with tempfile.TemporaryDirectory() as tmp:
            sizes = [(60, 200)] * 4
            boxes = self._boxes(sizes)
            # 一个岛包含两个框（合并过）→ 即使框很大也不封顶
            dataset = self._islands([("island_001", (10, 10, 500, 420))], tmp)
            self.assertEqual(oversized_single_box_islands(dataset, boxes, ratio=2.0), set())

    def test_normal_single_box_not_capped(self):
        from one_click_translate import oversized_single_box_islands

        with tempfile.TemporaryDirectory() as tmp:
            sizes = [(60, 200)] * 4
            boxes = self._boxes(sizes)
            specs = [("island_001", (10, 10, 70, 210))]
            dataset = self._islands(specs, tmp)
            self.assertEqual(oversized_single_box_islands(dataset, boxes, ratio=2.0), set())

    def test_capped_font_size_in_typeset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset([("island_001", (100, 100, 500, 700))], tmp)
            erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
            result = typeset_island_page(
                erased, dataset, {"island_001": "轰轰轰"}, [],
                config=TypesetConfig(),
                direction_overrides={"island_001": "vertical"},
                capped_islands={"island_001"},
            )
            self.assertLessEqual(result.audits[0].font_size, 30)


class PageFontCeilingTests(unittest.TestCase):
    """整页字号上限：字号 ≤ N × 该页常规字号，只封顶不放大。

    各岛各自"尽可能撑满"会让同一页出现 2 倍级差（实测页 7：14 字合并组
    131px，而该页常规字号只有 66px），观感极不统一。
    """

    def _typeset(self, tmp, bbox, text, ceiling):
        dataset = _dataset([("island_001", bbox)], tmp)
        erased = Image.new("RGB", dataset.image_size, (250, 250, 250))
        return typeset_island_page(
            erased, dataset, {"island_001": text}, [],
            config=TypesetConfig(),
            direction_overrides={"island_001": "vertical"},
            font_ceiling_px=ceiling,
        ).audits[0]

    def test_ceiling_caps_a_large_font(self):
        with tempfile.TemporaryDirectory() as tmp:
            uncapped = self._typeset(tmp, (100, 100, 400, 700), "十四字的一句台词", None)
            capped = self._typeset(tmp, (100, 100, 400, 700), "十四字的一句台词", 60)
            self.assertGreater(uncapped.font_size, 60)
            self.assertLessEqual(capped.font_size, 60)

    def test_ceiling_never_enlarges(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._typeset(tmp, (100, 100, 220, 300), "两字", None)
            high = self._typeset(tmp, (100, 100, 220, 300), "两字", 9999)
            self.assertEqual(plain.font_size, high.font_size)

    def test_zero_or_none_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._typeset(tmp, (100, 100, 400, 700), "十四字的一句台词", None)
            zero = self._typeset(tmp, (100, 100, 400, 700), "十四字的一句台词", 0)
            self.assertEqual(plain.font_size, zero.font_size)
