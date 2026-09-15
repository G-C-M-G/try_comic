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


class ArtTextBoxTests(unittest.TestCase):
    """手绘字（画在画面上的效果字/拟声词）在 join 阶段被整岛丢弃。

    这类岛的"原文"读出来是像模像样的一句话（实测「よろしく」「だいすき」
    「だめ」都是手绘字），语义判据分不出来——只有"岛底不平坦"（几何，见
    attach_bg_flatness）叠"原文很短"（字数）才分得开。压在画面上的印刷正文
    （封面警示语、后记段落）字数都超限，必须照旧入译。

    判据是**整岛**：同一句对白的几个框各读一截，逐框判会把印在气泡里的对白
    判成手绘字（实测过）。
    """

    def _units(self, box=None, **island_extra):
        island = {"id": "island_001", "image_bbox": [0, 0, 1000, 1000]}
        island.update(island_extra)
        return {
            "1": _unit(
                {"island_001": island},
                {"a": {"id": "a", "image_bbox": box or [100, 100, 300, 900],
                       "page_scale": False}},
                {"g1": {"id": "g1", "members": ["a"]}},
            )
        }

    def test_short_reading_on_art_is_dropped(self):
        units = self._units(bg_flat_ratio=0.08)
        texts, _ = build_page_texts(units, {"1:box:a": "よろしく"}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "")

    def test_long_horizontal_text_on_art_is_kept(self):
        """压在画面上的**横排印刷长句**（封面警示语、后记段落）照旧入译。

        竖排豁免只管竖排；横排的正文仍受字数护栏保护，否则封面警示语会被吃掉。
        """
        units = self._units(box=[100, 100, 900, 180], bg_flat_ratio=0.20)
        long_text = "この時期にエアコンぶっ壊れて死にそうになったけど"
        texts, _ = build_page_texts(units, {"1:box:a": long_text}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], long_text)

    def test_long_vertical_text_on_art_is_dropped(self):
        """竖排的长拟声不再被字数护栏放过（实测长拟声占比 0.15~0.37）。"""
        units = self._units(bg_flat_ratio=0.30)
        long_text = "びくっぷるぷるふるふるえる"
        texts, _ = build_page_texts(units, {"1:box:a": long_text}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "")

    def test_short_reading_on_balloon_is_kept(self):
        """气泡上的短对白（岛底平坦）照旧入译——这是判据最容易误伤的一类。"""
        units = self._units(bg_flat_ratio=0.90)
        texts, _ = build_page_texts(units, {"1:box:a": "んっ♡"}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "んっ♡")

    def test_island_without_flatness_measurement_is_kept(self):
        """老数据集没有 bg_flat_ratio → 判据无法自证 → 保守保留。"""
        units = self._units()
        texts, _ = build_page_texts(units, {"1:box:a": "よろしく"}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "よろしく")


class SameLetteringTests(unittest.TestCase):
    """「同一处字」判据（_same_lettering）：归一化后一短一长、短的落在长的开头。"""

    def _same(self, a, b):
        from one_click_translate import _same_lettering

        return _same_lettering(a, b)

    def test_same_lettering_read_differently(self):
        """实测的两例：同一团手绘字被两个重叠框读成不同长度。"""
        self.assertTrue(self._same("ぬるっと♡じゅっ", "ぬるっと♡じゅっ必ずい"))
        self.assertTrue(self._same("かわいい♥", "かわい♡"))     # 装饰符号不参与比较

    def test_unrelated_readings_are_not_the_same_lettering(self):
        self.assertFalse(self._same("えーと", "かわいい♡"))
        self.assertFalse(self._same("そこで", "であり"))

    def test_too_short_never_matches(self):
        """1 个字太容易巧合，不参与（"あ" 不该匹配 "ああ"）。"""
        self.assertFalse(self._same("あ", "ああ"))
        self.assertFalse(self._same("", "ぬるっと"))


class ArtTextOverlapTests(unittest.TestCase):
    """同一处字被切成两个重叠岛时同判。

    两个框盖着同一团手绘字时，矩形只差一点、占比却可能分居阈值两侧（实测
    页 17「かわいい♥」0.46 留 /「かわい♡」0.37 丢）。同判取**最小占比**（任一
    段明显落在画面上，整团就算画在画面上）+ **最长的一段原文**（取最短会把
    印刷长句的碎片误杀）。
    """

    def _units_with_boxes(self, left_ratio, right_ratio, *, island_a,
                          island_b, box_a, box_b):
        return self._units(left_ratio, right_ratio, box_a, box_b,
                           island_a=island_a, island_b=island_b)

    def _units(self, left_ratio, right_ratio, left_box, right_box,
               island_a=None, island_b=None):
        islands = {
            "island_001": {"id": "island_001",
                           "image_bbox": island_a or [0, 0, 200, 400],
                           "members_bbox": island_a or [0, 0, 200, 400],
                           "bg_flat_ratio": left_ratio},
            "island_002": {"id": "island_002",
                           "image_bbox": island_b or [160, 0, 360, 400],
                           "members_bbox": island_b or [160, 0, 360, 400],
                           "bg_flat_ratio": right_ratio},
        }
        blocks = {
            "a": {"id": "a", "image_bbox": left_box, "page_scale": False},
            "b": {"id": "b", "image_bbox": right_box, "page_scale": False},
        }
        groups = {"g1": {"id": "g1", "members": ["a"]},
                  "g2": {"id": "g2", "members": ["b"]}}
        return {"1": _unit(islands, blocks, groups)}

    def test_overlapping_same_lettering_share_the_verdict(self):
        units = self._units(0.46, 0.37, [20, 20, 180, 380], [180, 20, 340, 380])
        readings = {"1:box:a": "かわいい♥", "1:box:b": "かわい♡"}
        texts, _ = build_page_texts(units, readings, {"1": set()}, flat_gate=0.70)
        # 0.46 那个本来会留下，但和 0.37 的兄弟框是同一处字 → 一起丢
        self.assertEqual(texts["1"]["island_001"], "")
        self.assertEqual(texts["1"]["island_002"], "")

    def test_unrelated_readings_are_judged_separately(self):
        units = self._units(0.85, 0.37, [20, 20, 180, 380], [180, 20, 340, 380])
        readings = {"1:box:a": "えーと", "1:box:b": "かわいい♡"}
        texts, _ = build_page_texts(units, readings, {"1": set()}, flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "えーと")    # 气泡里 → 留
        self.assertEqual(texts["1"]["island_002"], "")

    def test_longest_reading_decides_the_length_guard(self):
        """同判取最长的一段：不能因为"短的那半 ≤10 字"就把长句误杀。

        用**横排**（竖排低占比直接丢，测不到护栏这一层）。
        """
        # 两个横排框上下相邻、岛矩形相接（能触发"同一处字"归并），
        # 但每个岛只装得下自己那个框（中心判定）
        units = self._units_with_boxes(
            0.30, 0.30,
            island_a=[0, 0, 340, 200], island_b=[0, 200, 340, 400],
            box_a=[20, 20, 320, 60], box_b=[20, 220, 320, 260])
        readings = {"1:box:a": "ぬるっとじゅっ",
                    "1:box:b": "ぬるっと♡じゅっ必ずい"}   # 11 字（含♡）> 10
        texts, _ = build_page_texts(units, readings, {"1": set()}, flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], readings["1:box:a"])


class CommentaryReadingTests(unittest.TestCase):
    """AI 说明性输出按空读数处理——否则它会用字数把护栏骗过去。

    实测页 4 一个岛：占比 0.23 本该判成手绘字，但它的读数是视觉模型的一句
    说明（「图片中无任何可识别的日文文本…」37 字），撑爆 ≤10 字的护栏，于是
    整岛被当成"印刷长文"保留。
    """

    def test_commentary_reading_is_treated_as_empty(self):
        island = {"id": "island_001", "image_bbox": [0, 0, 200, 400],
                  "members_bbox": [0, 0, 200, 400], "bg_flat_ratio": 0.10}
        units = {"1": _unit(
            {"island_001": island},
            {"a": {"id": "a", "image_bbox": [20, 20, 180, 380],
                   "page_scale": False}},
            {"g1": {"id": "g1", "members": ["a"]}},
        )}
        readings = {"1:box:a": "图片中无任何可识别的日文文本，仅含手绘爱心轮廓"
                               "与网点纹理，无法输出文字。"}
        texts, _ = build_page_texts(units, readings, {"1": set()}, flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "")

    def test_real_reading_still_passes_through(self):
        island = {"id": "island_001", "image_bbox": [0, 0, 200, 400],
                  "members_bbox": [0, 0, 200, 400], "bg_flat_ratio": 0.90}
        units = {"1": _unit(
            {"island_001": island},
            {"a": {"id": "a", "image_bbox": [20, 20, 180, 380],
                   "page_scale": False}},
            {"g1": {"id": "g1", "members": ["a"]}},
        )}
        texts, _ = build_page_texts(units, {"1:box:a": "んっ♡"}, {"1": set()},
                                    flat_gate=0.70)
        self.assertEqual(texts["1"]["island_001"], "んっ♡")


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
        # 字号上限取**未裁剪**的扩框尺寸（与岛 bbox 解耦）：A = 40+12 = 52，
        # B = 60+12 = 72 → min = 52。掩膜仍用被岛裁剪的局部坐标（见下面
        # area 断言），但字号上限不该随岛 bbox 变大而变大。
        self.assertEqual(overrides["island_001"]["glyph_width_limit"], 52.0)
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
    def test_vertical_side_by_side_top_aligned_boxes_merge(self):
        """竖排的合并判据 = **顶端对齐**（≤6px）+ 横向够近（<20% 较小框宽）。"""
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 80, 30, 110, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_top_misaligned_boxes_do_not_merge(self):
        """顶端差 30px、纵向大片重叠，也不算一组（旧判据会并，用户实书反馈后改）。"""
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 80, 60, 110, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_length_mismatch_but_top_aligned_merges(self):
        """长列 + 短列：重叠只有长列的 27%，但顶端对齐 → 同一句话，并。

        这是新判据要修的那类（旧判据"重叠 > 较大框 40%"会漏掉它）。
        """
        boxes = [
            _box("a", 50, 30, 80, 430),     # 400 高
            _box("b", 85, 31, 115, 140),    # 110 高，顶端只差 1px
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_narrow_columns_within_ratio_merge(self):
        # 窄列（20px 宽）间距 5px < 20% × 20px = 4px？不——5 > 4，改 3px。
        # 顶端都在 64，横向 3px < 4px -> 并。
        boxes = [
            _box("a", 1084, 64, 1104, 220),
            _box("b", 1107, 64, 1130, 174),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_9px_gap_no_longer_merges_at_default_ratio(self):
        # 同样的窄列把间距放到 9px：9 > 6 -> 不并。记录默认比例收紧后的边界
        # （旧规则在扩边后几何上判，扩边白送 2×pad，等价阈值宽松得多）。
        boxes = [
            _box("a", 1084, 64, 1104, 220),
            _box("b", 1113, 64, 1136, 174),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 2)

    def test_vertical_gap_within_ratio_merges(self):
        # 合并判据在**原始框**上：顶端对齐 + 间距 5px < 20% × 较小框宽 (30px → 6px)。
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 85, 31, 115, 220),
        ]
        groups = _group_boxes_by_overlap(boxes)
        self.assertEqual(len(groups), 1)

    def test_vertical_gap_beyond_ratio_does_not_merge(self):
        # 同一对框把间距拉到 15px：15 > 30% × 30px = 9px -> 不并。
        boxes = [
            _box("a", 50, 30, 80, 200),
            _box("b", 95, 40, 125, 220),
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


class ArtTextGateTests(unittest.TestCase):
    """按书自标定闸门 + 自检（art_text_gate）。

    用这本书自己的长文本岛（≥12 字，几乎必然是对白）自证"气泡基准"：
    基准又高又稳才敢用几何判据，否则**整本禁用**（宁可全留）。
    实测（长文本岛 p25）：甲书 0.76 → 可用（闸门 0.69，岛级 F1 0.985）；
    乙书 0.56 → 自证不住，强跑精确率只有 30.9%。
    """

    def _units(self, ratios):
        islands, blocks, groups = {}, {}, {}
        for index, ratio in enumerate(ratios, 1):
            iid, bid = f"island_{index:03d}", f"b{index}"
            islands[iid] = {"id": iid, "image_bbox": [0, 0, 200, 400],
                            "members_bbox": [0, 0, 200, 400],
                            "bg_flat_ratio": ratio}
            blocks[bid] = {"id": bid, "image_bbox": [20, 20, 180, 380],
                           "page_scale": False}
            groups[f"g{index}"] = {"id": f"g{index}", "members": [bid]}
        return {"1": _unit(islands, blocks, groups)}

    def _texts(self, count, length, short=None):
        """前 count 个岛给 length 字；short = (起, 止, 字数) 用来标"短岛"。"""
        texts = {f"island_{i:03d}": "あ" * length for i in range(1, count + 1)}
        if short:
            first, last, size = short
            for i in range(first, last + 1):
                texts[f"island_{i:03d}"] = "あ" * size
        return {"1": texts}

    def test_gate_derived_from_long_text_islands(self):
        """长文本岛占比高而稳 → 闸门 ≈ 0.9 × 中位（甲书：0.77 → 0.69）。"""
        from one_click_translate import art_text_gate

        # 10 个长文本岛（对白）占比 0.77，另 2 个短岛（拟声）占比 0.20/0.25
        units = self._units([0.77] * 10 + [0.20, 0.25])
        gate, basis = art_text_gate(units, self._texts(12, 40, short=(11, 12, 3)))
        self.assertIsNotNone(gate)
        self.assertAlmostEqual(gate, 0.69, places=2)
        self.assertIn("0.77", basis)

    def test_gate_disabled_when_baseline_self_certifies_poorly(self):
        """长文本岛占比铺得开、p25 低于 0.65 → 禁用（乙书 的真实形状）。"""
        from one_click_translate import art_text_gate

        # 乙书 实测：长文本岛中位 0.67 / p25 0.56——对白自己就有一半落在
        # 几何闸门下，这种书上判据没有信号（强跑精确率 30.9%）。
        units = self._units([0.40, 0.46, 0.50, 0.53, 0.55, 0.56, 0.60, 0.62,
                             0.64, 0.66, 0.68, 0.70, 0.70, 0.71, 0.72, 0.72,
                             0.73, 0.74, 0.75, 0.76])

        gate, basis = art_text_gate(units, self._texts(20, 40))
        self.assertIsNone(gate)
        self.assertIn("自证不住", basis)

    def test_few_low_islands_do_not_disable_a_good_book(self):
        """少数低占比长文本岛不该决定整本判定（用 p25 而不是 p10 的原因）。

        长文本岛里必然混着少数"不该算气泡基准"的岛：横排印刷长文（封面警示语
        0.17 / 后记段落 0.34）与偶然被读出 ≥12 字的手绘字（ぬるっと♡じゅっ必ずい
        0.15）。甲书 一本在当前编组下 p10 是 0.67、换成老编组就变 0.37——判据
        没动、只是编组变了。140 个长文本岛里 20 个污染（14%）就是 p10 与 p25 的
        分界：p10 被污染岛占住（→ 会误禁用），p25 仍然落在对白簇里。
        """
        from one_click_translate import art_text_gate

        units = self._units([0.15] * 20 + [0.77] * 120)
        gate, basis = art_text_gate(units, self._texts(140, 40))
        self.assertIsNotNone(gate)
        self.assertAlmostEqual(gate, 0.69, places=2)
        self.assertIn("p25", basis)

    def test_gate_disabled_when_samples_too_few(self):
        from one_click_translate import art_text_gate

        units = self._units([0.9, 0.9])
        gate, basis = art_text_gate(units, self._texts(2, 40))
        self.assertIsNone(gate)
        self.assertIn("样本不足", basis)

    def test_short_islands_do_not_count_as_baseline(self):
        """只有长文本岛才算气泡基准（短岛可能本身就是拟声）。"""
        from one_click_translate import art_text_gate

        units = self._units([0.20] * 12)
        gate, _ = art_text_gate(units, self._texts(12, 3))
        self.assertIsNone(gate)

    def test_disabled_book_drops_nothing(self):
        """禁用时一个岛都不丢（全部照常翻译）。"""
        units = self._units([0.10] * 10)
        readings = {f"1:box:b{i}": "ぎゅっ♡" for i in range(1, 11)}
        texts, _ = build_page_texts(units, readings, {"1": set()})
        self.assertTrue(all(texts["1"][key] for key in texts["1"]))


class VisionJudgeTests(unittest.TestCase):
    """自证不住的书改用视觉判据（S/D/U）—— 接管线这一段。

    三档：S 拟声 → 不译；D 台词 → 照常译；U 图里没有可读的字 → 不译。
    """

    def _units(self, ratios=(0.30, 0.35), texts=("ぎゅっ", "ぎゅっ")):
        """每个岛竖着隔开，保证"框中心落在岛内"只命中自己那一个框。"""
        islands, blocks, groups = {}, {}, {}
        for index, ratio in enumerate(ratios, 1):
            iid, bid = f"island_{index:03d}", f"b{index}"
            top = (index - 1) * 500
            islands[iid] = {"id": iid, "image_bbox": [0, top, 200, top + 400],
                            "members_bbox": [0, top, 200, top + 400],
                            "bg_flat_ratio": ratio}
            blocks[bid] = {"id": bid, "image_bbox": [20, top + 20, 180, top + 380],
                           "page_scale": False}
            groups[f"g{index}"] = {"id": f"g{index}", "members": [bid]}
        return {"1": _unit(islands, blocks, groups)}, {
            f"1:box:b{i}": t for i, t in enumerate(texts, 1)}

    def test_feature_off_never_calls_the_endpoint(self):
        """数据集没算平坦度（功能没开）→ 绝不调接口。"""
        units, readings = self._units()
        for island in units["1"]["islands"].values():
            island.pop("bg_flat_ratio")
        calls = []
        build_page_texts(units, readings, {},
                         vision_judge=lambda u, t: calls.append(1) or set())
        self.assertEqual(calls, [])

    def test_self_check_failure_routes_to_vision(self):
        """自检不住（没有长文本岛可自证）→ 走视觉判据；S/U 丢、D 留。"""
        units, readings = self._units(ratios=(0.30, 0.30, 0.30),
                                      texts=("ぎゅっ", "ぎゅっ", "ぎゅっ"))
        seen = {}

        def judge(units_arg, texts_arg):
            seen["texts"] = texts_arg
            return {"1:island_001", "1:island_003"}      # S 与 U

        texts, _ = build_page_texts(units, readings, {}, vision_judge=judge)
        self.assertEqual(texts["1"]["island_001"], "")
        self.assertEqual(texts["1"]["island_002"], "ぎゅっ")
        self.assertEqual(texts["1"]["island_003"], "")

    def test_self_certifying_book_does_not_call_vision(self):
        """自证得住的书（长文本岛又多又稳）→ 走几何判据，不调接口。"""
        islands = {}
        for index in range(1, 11):
            iid, bid = f"island_{index:03d}", f"b{index}"
            islands[iid] = {"id": iid, "image_bbox": [0, 0, 200, 400],
                            "members_bbox": [0, 0, 200, 400],
                            "bg_flat_ratio": 0.77}
        unit = _unit(islands,
                     {f"b{i}": {"id": f"b{i}", "image_bbox": [20, 20, 180, 380],
                                "page_scale": False} for i in range(1, 11)},
                     {f"g{i}": {"id": f"g{i}", "members": [f"b{i}"]}
                      for i in range(1, 11)})
        readings = {f"1:box:b{i}": "あ" * 20 for i in range(1, 11)}
        calls = []
        build_page_texts({"1": unit}, readings, {},
                         vision_judge=lambda u, t: calls.append(1) or set())
        self.assertEqual(calls, [])


class SfxJudgeModuleTests(unittest.TestCase):
    """`manga_localizer.sfx_judge`：三档解析 + 失败隔离。"""

    def _crop(self, root, name="1.jpg"):
        from PIL import Image

        path = Path(root) / name
        Image.new("RGB", (60, 60), "white").save(path)
        return {"id": "1:island_001:0,0,20,20", "source": str(path),
                "bbox": [0, 0, 20, 20], "pad": 5}

    def test_parses_s_d_u(self):
        from unittest import mock

        from manga_localizer import sfx_judge

        with tempfile.TemporaryDirectory() as tmp:
            crop = self._crop(tmp)
            with mock.patch.object(sfx_judge, "_ask", return_value="U"):
                self.assertEqual(sfx_judge.judge_crops([crop], "u", "m", "k"),
                                 {crop["id"]: "U"})
            with mock.patch.object(sfx_judge, "_ask", return_value="S"):
                self.assertEqual(sfx_judge.judge_crops([crop], "u", "m", "k"),
                                 {crop["id"]: "S"})

    def test_unparsable_answer_is_empty(self):
        from unittest import mock

        from manga_localizer import sfx_judge

        with tempfile.TemporaryDirectory() as tmp:
            crop = self._crop(tmp)
            with mock.patch.object(sfx_judge, "_ask",
                                   return_value="这张图没有清晰可读的日文"):
                self.assertEqual(sfx_judge.judge_crops([crop], "u", "m", "k"),
                                 {crop["id"]: ""})

    def test_endpoint_error_is_isolated(self):
        from unittest import mock

        from manga_localizer import sfx_judge

        with tempfile.TemporaryDirectory() as tmp:
            crop = self._crop(tmp)
            with mock.patch.object(sfx_judge, "_ask",
                                   side_effect=RuntimeError("boom")):
                self.assertEqual(sfx_judge.judge_crops([crop], "u", "m", "k"),
                                 {crop["id"]: ""})
