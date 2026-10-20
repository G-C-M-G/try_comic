from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from manga_localizer.errors import TypesettingError
from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_dry_run import OcrLine
from manga_localizer.island_typeset import (
    PROHIBITED_LINE_END,
    PROHIBITED_LINE_START,
    TypesetConfig,
    _block_metrics,
    _slots_centered,
    _slots_follow_columns,
    _slots_horizontal,
    apply_kinsoku,
    kinsoku_violations,
    typeset_island_page,
)


def _line(x1, y1, x2, y2, text="x", island_id=None):
    line = OcrLine(polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)), text=text)
    if island_id:
        object.__setattr__(line, "island_id", island_id)
    return line


def _dataset(islands, size=(300, 400)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


def _island(bbox, area=9000.0, island_id="island_001"):
    return {
        "id": island_id,
        "image_bbox": list(bbox),
        "image_center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
        "image_area": area,
    }


class KinsokuTest(unittest.TestCase):
    """禁则（避头尾）：只移动断点，不动字序、不超上限。"""

    def test_violations_counted(self):
        # 「，」落在第二行行首
        self.assertEqual(kinsoku_violations("甲乙丙，丁戊", [3, 3]), 1)
        # 「（」落在第一行行末
        self.assertEqual(kinsoku_violations("甲乙（丙丁戊", [3, 3]), 1)
        self.assertEqual(kinsoku_violations("甲乙丙丁戊己", [3, 3]), 0)

    def test_first_line_may_start_with_punctuation(self):
        """整句以「…」起是原文自己的开头，不算断行违规。"""
        self.assertEqual(kinsoku_violations("…甲乙丙丁戊", [3, 3]), 0)

    def test_line_end_opening_bracket_counted_at_text_end(self):
        self.assertEqual(kinsoku_violations("甲乙丙丁戊（", [3, 3]), 1)

    def test_shifts_break_to_avoid_line_start_punctuation(self):
        """「，」被挤到第二列行首 → 往前挪断点，它落到上一列末尾。"""
        text = "甲乙丙，丁戊己"
        counts = apply_kinsoku(text, [3, 3, 1], 3)
        self.assertEqual(kinsoku_violations(text, counts), 0)
        self.assertEqual(sum(counts), len(text))
        self.assertEqual(len(counts), 3)

    def test_shifts_break_to_avoid_line_end_bracket(self):
        """「（」落在第一列末尾 → 往前挪断点，它变成第二列行首（合法）。"""
        text = "甲乙（丙丁戊己"
        counts = apply_kinsoku(text, [3, 3, 1], 3)
        self.assertEqual(kinsoku_violations(text, counts), 0)
        self.assertEqual(sum(counts), len(text))

    def test_invariants_hold(self):
        """行数不变、总和不变、每行字数仍在 [1, capacity]。"""
        text = "彼は言った、それは違う。本当にそう思う、なぜなら"
        capacity = 4
        base = [capacity] * (len(text) // capacity) + [len(text) % capacity]
        base = [c for c in base if c > 0]
        counts = apply_kinsoku(text, base, capacity)
        self.assertEqual(len(counts), len(base))
        self.assertEqual(sum(counts), len(text))
        self.assertTrue(all(1 <= c <= capacity for c in counts))

    def test_never_worse_than_input(self):
        """不许把本来正确的改坏，也不许把违规数改多。"""
        for text, counts in (
            ("甲乙丙丁戊己", [3, 3]),
            ("甲乙、丙丁戊", [3, 3]),
            ("（甲乙丙丁戊", [3, 3]),
            ("甲乙丙、丁戊己庚", [4, 4]),
            ("，，，，，，", [3, 3]),
        ):
            out = apply_kinsoku(text, counts, max(counts))
            self.assertLessEqual(kinsoku_violations(text, out),
                                 kinsoku_violations(text, counts))
            self.assertEqual(sum(out), len(text))

    def test_full_lines_without_slack_left_alone(self):
        """每行都顶满 = 没有空位可借 → 保持原样。

        这是本实现的**有意边界**：宁可留一处违规，也不让某行超上限（那会让字
        跑出岛外）。真实布局里末行是余数、总有余位，所以能修好的占多数。
        """
        text = "甲乙丙、。丁"
        counts = apply_kinsoku(text, [3, 3], 3)
        self.assertEqual(counts, (3, 3))

    def test_single_line_untouched(self):
        self.assertEqual(apply_kinsoku("甲乙丙，丁", [5], 5), (5,))

    def test_slots_centered_respects_kinsoku(self):
        dataset = _dataset([_island((10, 10, 200, 300))])
        island = dataset.islands[0]
        text = "甲乙丙，丁戊己庚辛壬癸"
        slots = _slots_centered(text, 24, island, (105, 155), TypesetConfig())
        counts = [slot.char_count for slot in slots]
        self.assertEqual(sum(counts), len(text))
        self.assertEqual(kinsoku_violations(text, counts), 0)

    def test_slots_horizontal_respects_kinsoku(self):
        dataset = _dataset([_island((10, 10, 290, 160))])
        island = dataset.islands[0]
        text = "甲乙丙丁戊，己庚辛壬癸子丑"
        slots = _slots_horizontal(text, 18, island, (150, 85), TypesetConfig())
        counts = [slot.char_count for slot in slots]
        self.assertEqual(sum(counts), len(text))
        self.assertEqual(kinsoku_violations(text, counts), 0)

    def test_prohibited_sets_are_disjoint(self):
        """避头集与避尾集互不相交——否则一处标点两个方向都无解。"""
        self.assertEqual(PROHIBITED_LINE_START & PROHIBITED_LINE_END, frozenset())


class ReadingOrderTest(unittest.TestCase):
    def test_follow_columns_right_to_left(self):
        dataset = _dataset([_island((10, 10, 200, 300))])
        island = dataset.islands[0]
        columns = [
            type("C", (), {"bbox": (150, 40, 170, 280), "direction": "vertical", "lines": []})(),
            type("C", (), {"bbox": (110, 40, 130, 280), "direction": "vertical", "lines": []})(),
        ]
        slots = _slots_follow_columns("甲乙丙丁", 24, columns, island, TypesetConfig())
        self.assertEqual(len(slots), 1)
        # first (reading) column must sit on the RIGHTMOST original column
        self.assertAlmostEqual(slots[0].x_center, 160.0, places=3)

    def test_centered_columns_right_to_left(self):
        dataset = _dataset([_island((10, 10, 200, 300))])
        island = dataset.islands[0]
        slots = _slots_centered(
            "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未", 24, island, (105, 155), TypesetConfig()
        )
        self.assertGreater(len(slots), 1)
        xs = [slot.x_center for slot in slots]
        # column 0 is rightmost, subsequent columns step leftwards
        self.assertGreater(xs[0], xs[1])


class MinRatioTest(unittest.TestCase):
    def test_default_min_ratio_is_0_6(self):
        self.assertEqual(TypesetConfig().min_ratio, 0.6)

    def test_min_ratio_configurable(self):
        self.assertEqual(TypesetConfig(min_ratio=0.5).min_ratio, 0.5)

    def test_min_font_size_floor_is_28(self):
        self.assertEqual(TypesetConfig().min_font_size, 28)


class AlignmentRulesTest(unittest.TestCase):
    def test_vertical_columns_top_aligned(self):
        # Two columns of one sentence share the same top edge.
        dataset = _dataset([_island((10, 10, 200, 300))])
        island = dataset.islands[0]
        slots = _slots_centered(
            "甲乙丙丁戊己", 30, island, (105, 155), TypesetConfig(),
            layout_bbox=(10, 10, 200, 300), top_align=12,
        )
        step = 30 + 30 // 16
        expected_tops = [
            slot.y_center - ((slot.char_count - 1) * step + 30) / 2.0
            for slot in slots
        ]
        self.assertEqual(len(set(round(top, 3) for top in expected_tops)), 1)
        self.assertAlmostEqual(expected_tops[0], 12.0, places=3)

    def test_horizontal_rows_left_aligned(self):
        # A short last row starts at the same left edge as the full rows.
        dataset = _dataset([_island((10, 10, 300, 200))])
        island = dataset.islands[0]
        slots = _slots_horizontal(
            "甲乙丙丁戊己", 24, island, (155, 105), TypesetConfig(),
            layout_bbox=(10, 10, 300, 200),
        )
        step = 24 + 24 // 16
        lefts = [
            slot.x_center - (slot.char_count - 1) * step / 2.0
            for slot in slots
        ]
        self.assertEqual(len(set(round(left, 3) for left in lefts)), 1)


class TypesetPageTest(unittest.TestCase):
    def _page(self):
        islands = [
            _island((10, 10, 110, 210), area=9000.0, island_id="island_001"),
            _island((150, 10, 190, 210), area=4000.0, island_id="island_002"),
        ]
        return _dataset(islands)

    def test_typeset_and_zero_outside_change(self):
        dataset = self._page()
        image = Image.new("RGB", (300, 400), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for x in range(20, 100, 6):
            draw.line((x, 30, x, 200), fill=(0, 0, 0), width=2)
        translations = {"island_001": "这是一段测试文字", "island_002": "测试文字"}
        paddle = [
            _line(16, 14, 34, 206, island_id="island_001"),
            _line(42, 14, 60, 206, island_id="island_001"),
            _line(68, 14, 86, 206, island_id="island_001"),
            _line(152, 14, 170, 206, island_id="island_002"),
        ]
        result = typeset_island_page(
            image, dataset, translations, paddle, config=TypesetConfig()
        )
        audits = {item.island_id: item for item in result.audits}
        # Both texts fit entirely inside their island bboxes (the
        # canonical "one box") -> clean typeset.
        self.assertEqual(audits["island_001"].status, "typeset")
        self.assertEqual(audits["island_002"].status, "typeset")
        self.assertFalse(any("soft cap" in reason for reason in audits["island_001"].reasons))
        self.assertGreaterEqual(audits["island_001"].area_ratio, 0.6)
        self.assertLessEqual(audits["island_001"].area_ratio, 1.1)
        self.assertLessEqual(audits["island_002"].center_offset, audits["island_002"].center_tolerance)
        self.assertGreater(audits["island_001"].font_size, 0)
        # Canonical "one box" contract: the block never leaves the island
        # bbox (the erasure range covers the whole island).
        for audit in audits.values():
            bb = audit.block_bbox
            if bb is None:
                continue
            ix1, iy1, ix2, iy2 = next(
                i.image_bbox for i in dataset.islands if i.id == audit.island_id
            )
            self.assertGreaterEqual(bb[0], ix1)
            self.assertGreaterEqual(bb[1], iy1)
            self.assertLessEqual(bb[2], ix2)
            self.assertLessEqual(bb[3], iy2)

    def test_missing_translation_skipped(self):
        dataset = self._page()
        image = Image.new("RGB", (300, 400), (255, 255, 255))
        result = typeset_island_page(
            image, dataset, {"island_001": "测试"}, [], config=TypesetConfig()
        )
        by_id = {item.island_id: item for item in result.audits}
        self.assertEqual(by_id["island_002"].status, "skipped")

    def test_untranslatable_island_failed_audit(self):
        dataset = _dataset([_island((10, 10, 30, 30), area=400.0)])
        image = Image.new("RGB", (300, 400), (255, 255, 255))
        translations = {"island_001": "这是一段非常长的测试文字"}
        result = typeset_island_page(image, dataset, translations, [], config=TypesetConfig())
        # The text cannot fit even at the absolute floor (12px), but it is
        # still rendered (clipped to the island) so the island is never
        # blank: the status is pending_review, not translation_failed.
        self.assertEqual(result.audits[0].status, "typeset_pending_review")
        self.assertEqual(result.audits[0].font_size, 12)


if __name__ == "__main__":
    unittest.main()


class StrokeOverflowTest(unittest.TestCase):
    """描边是向外画的：裁剪掩膜必须按 STROKE_OVERFLOW_MARGIN 放宽。

    否则贴着布局框排的最外圈描边会被"岛外零改动"的硬裁剪切掉 —— 表现就是
    边缘的字缺一笔。
    """

    def test_stroke_near_box_edge_survives(self):
        from manga_localizer.thresholds import STROKE_OVERFLOW_MARGIN

        dataset = _dataset([_island((40, 40, 120, 260), area=8000.0)])
        image = Image.new("RGB", (300, 400), (255, 255, 255))
        translations = {"island_001": "测试文字"}
        result = typeset_island_page(
            image, dataset, translations, (), config=TypesetConfig()
        )
        island = dataset.islands[0]
        x1, y1, x2, y2 = island.image_bbox
        # 掩膜放宽带内允许落墨（不会因裁剪丢笔画）；再往外必须零改动
        margin = STROKE_OVERFLOW_MARGIN
        changed = result.changed_mask
        outside = [
            (x, y)
            for x in range(max(0, x1 - margin - 6), x1 - margin)
            for y in range(y1, y2)
            if changed.getpixel((x, y))
        ]
        self.assertEqual(outside, [], "放宽带之外不应有改动")
        self.assertGreater(result.audits[0].font_size, 0)
