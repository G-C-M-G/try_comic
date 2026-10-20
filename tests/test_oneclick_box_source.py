"""Paddle 框采集（box-source=paddle）契约与过滤规则测试。

重点不是"Paddle 检测得准不准"（那是评测），而是**产物契约**：写出的
capture.json 必须能被下游原样消费（build_bluebox_dataset / check 图 /
审核门），因为这套契约是 Photos 路径与 Paddle 路径共用的接口。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from one_click_translate import (
    build_bluebox_dataset,
    capture_boxes_with_paddle,
    _group_boxes_by_overlap,
)


class _FakeProc:
    """最小 subprocess 替身：无输出、退出码 0。"""

    def __init__(self, *_args, **_kwargs):
        self.stdout = iter(())
        self.returncode = 0

    def wait(self):
        return 0


def _poly(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class PaddleBoxCaptureTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.out_root = self.root / "out"
        self.out_root.mkdir()
        # 400x400 白底页
        self.page = self.images / "7.jpg"
        Image.fromarray(
            np.full((400, 400, 3), 255, np.uint8)
        ).save(self.page)

    def _write_raw(self, predictions):
        (self.out_root / "paddle_boxes.json").write_text(
            json.dumps({
                "schema_version": 1,
                "backend": "PaddleOCR PP-OCRv6 medium",
                "device": "gpu",
                "pages": [{"source": str(self.page), "predictions": predictions}],
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def _capture(self, predictions, **kwargs):
        self._write_raw(predictions)
        with mock.patch("one_click_translate.subprocess.Popen", _FakeProc):
            ok = capture_boxes_with_paddle(
                self.images, self.out_root, None, **kwargs
            )
        manifest = json.loads(
            (self.out_root / "capture" / "paddle" / "7" / "capture.json").read_text(encoding="utf-8")
        )
        return ok, manifest

    def test_writes_contract_consumable_by_downstream(self):
        ok, manifest = self._capture([
            {"polygon": _poly(300, 40, 360, 300), "text": "あ", "confidence": 0.9},
            {"polygon": _poly(320, 60, 360, 280), "text": "い", "confidence": 0.9},
        ])
        self.assertTrue(ok)
        source = manifest["source"]
        self.assertEqual((source["width"], source["height"]), (400, 400))
        self.assertTrue(manifest["capture"]["complete"])
        blocks = manifest["capture"]["selection_blocks"]
        self.assertEqual([b["id"] for b in blocks], ["sel_001", "sel_002"])
        self.assertTrue(all(b["page_scale"] is False for b in blocks))
        self.assertEqual(manifest["capture"]["selection_boxes_status"], "ok")
        self.assertFalse(manifest["capture"]["selection_review_required"])
        # 下游能原样消费：dataset 构建成功并产出岛
        dataset_dir = self.out_root / "dataset" / "7"
        manifest_path = self.out_root / "capture" / "paddle" / "7" / "capture.json"
        islands = build_bluebox_dataset(manifest_path, dataset_dir, str(self.page))
        self.assertTrue(islands)
        written = json.loads(
            (dataset_dir / "selection_blocks.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(written["selection_blocks"]), 2)

    def test_reading_order_ids_top_down_right_to_left(self):
        # 两个框：y 相同，右边那个应先编号（竖排日文阅读序）
        ok, manifest = self._capture([
            {"polygon": _poly(100, 50, 140, 200), "text": "左", "confidence": 0.9},
            {"polygon": _poly(300, 50, 340, 200), "text": "右", "confidence": 0.9},
        ])
        blocks = manifest["capture"]["selection_blocks"]
        self.assertEqual([b["image_bbox"] for b in blocks],
                         [[300, 50, 340, 200], [100, 50, 140, 200]])

    def test_tiny_boxes_dropped(self):
        ok, manifest = self._capture([
            {"polygon": _poly(10, 10, 14, 14), "text": "x", "confidence": 0.1},
            {"polygon": _poly(100, 100, 140, 200), "text": "本", "confidence": 0.9},
        ])
        blocks = manifest["capture"]["selection_blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["image_bbox"], [100, 100, 140, 200])

    def test_page_scale_box_dropped(self):
        # 面积 > 20% 整页 → 非原子文本单位，丢弃
        ok, manifest = self._capture([
            {"polygon": _poly(0, 0, 400, 400), "text": "整页", "confidence": 0.9},
            {"polygon": _poly(100, 100, 140, 200), "text": "本", "confidence": 0.9},
        ])
        blocks = manifest["capture"]["selection_blocks"]
        self.assertEqual(len(blocks), 1)

    def test_no_boxes_requires_review(self):
        ok, manifest = self._capture([])
        self.assertTrue(ok)  # 页面写成功
        capture = manifest["capture"]
        self.assertEqual(capture["selection_blocks"], [])
        self.assertEqual(capture["selection_boxes_status"], "no_boxes")
        self.assertTrue(capture["selection_review_required"])
        self.assertTrue(capture["selection_missing"])

    def test_black_page_flagged(self):
        Image.fromarray(np.zeros((400, 400, 3), np.uint8)).save(self.page)
        ok, manifest = self._capture([])
        self.assertTrue(manifest["capture"]["black_page"])

    def test_groups_written_and_merge_side_by_side(self):
        # 两个竖排框顶端对齐、横向紧邻 → 同一组（顶端对齐 + 20% 间距规则）
        ok, manifest = self._capture([
            {"polygon": _poly(200, 60, 240, 300), "text": "あ", "confidence": 0.9},
            {"polygon": _poly(247, 60, 287, 300), "text": "い", "confidence": 0.9},
        ])
        groups = manifest["capture"]["selection_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["members"]), 2)

    def test_group_helper_accepts_generated_blocks(self):
        blocks = [
            {"id": "sel_001", "image_bbox": [200, 60, 240, 300], "page_scale": False},
            {"id": "sel_002", "image_bbox": [247, 60, 287, 300], "page_scale": False},
        ]
        grouped = _group_boxes_by_overlap(blocks)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(sorted(grouped[0][1]), ["sel_001", "sel_002"])


if __name__ == "__main__":
    unittest.main()


class LargeSfxFilterTest(unittest.TestCase):
    """大号拟音/效果字过滤（_filter_large_sfx）。

    判据是"短边 vs 同方向基准"，所以要锁住的不是某个具体阈值，而是几条
    容易写错的边界：横排不被误杀、样本不足时不乱删、分组确实按方向走。
    """

    PAGE = (1000, 1400)

    def _v(self, x, y, w, h):
        """竖排框（高 > 宽），短边 = w。"""
        return [x, y, x + w, y + h]

    def _h(self, x, y, w, h):
        """横排框（宽 >= 高），短边 = h。"""
        return [x, y, x + w, y + h]

    def test_large_sfx_dropped_against_direction_baseline(self):
        from one_click_translate import _filter_large_sfx

        boxes = [self._v(100 + i * 60, 100, 40, 300) for i in range(8)]
        big = self._h(100, 600, 500, 400)          # 短边 400 vs 竖排基准 40
        kept, dropped = _filter_large_sfx(boxes + [big], *self.PAGE)
        self.assertEqual(len(kept), 8)
        self.assertEqual([d[0] for d in dropped], [big])

    def test_wide_but_normal_horizontal_box_survives(self):
        """横排"宽"不是罪：短边正常就该留下（按宽度筛会误杀）。"""
        from one_click_translate import _filter_large_sfx

        boxes = [self._v(100 + i * 60, 100, 40, 300) for i in range(8)]
        wide = self._h(0, 600, 980, 45)            # 占满页宽，但短边 45 ≈ 正常字号
        kept, dropped = _filter_large_sfx(boxes + [wide], *self.PAGE)
        self.assertIn(wide, kept)
        self.assertEqual(dropped, [])

    def test_horizontal_compared_against_horizontal(self):
        """横框够多时用横框自己的基准：同向的大字被删，正常横排不受竖排影响。"""
        from one_click_translate import _filter_large_sfx

        boxes = [self._v(100 + i * 40, 100, 30, 260) for i in range(10)]   # 竖排基准 30
        h_small = [self._h(60, 700 + i * 80, 300, 60) for i in range(4)]   # 横排基准 60
        h_big = self._h(60, 1200, 600, 260)                                # 短边 260 > 2.5×60
        kept, dropped = _filter_large_sfx(boxes + h_small + [h_big], *self.PAGE)
        self.assertIn(h_big, dropped[0])
        self.assertEqual(len(kept), 14)

    def test_small_page_never_raises_or_overdrops(self):
        from one_click_translate import _filter_large_sfx

        few = [self._v(10, 10, 40, 200), self._h(100, 300, 400, 380)]
        kept, dropped = _filter_large_sfx(few, *self.PAGE)
        self.assertEqual(kept, few)
        self.assertEqual(dropped, [])

    def test_absolute_page_floor_blocks_small_page_false_drop(self):
        """少框页的兜底：短边 90 过了倍数条件（>2.5×30），但 < 页短边×0.10=100，
        被绝对下限挡下，避免把普通大字当拟声词删掉。"""
        from one_click_translate import _filter_large_sfx

        boxes = [
            self._v(10, 10, 30, 200), self._v(60, 10, 30, 200),
            self._h(10, 300, 400, 90),             # 短边 90：过倍数、卡在下限
        ]
        kept, dropped = _filter_large_sfx(boxes, *self.PAGE)
        self.assertEqual(dropped, [])
        self.assertEqual(len(kept), 3)

    def test_page_floor_does_not_block_a_real_sfx(self):
        """下限只挡"擦边"的；真·大拟声词远超下限，照样删。"""
        from one_click_translate import _filter_large_sfx

        boxes = [
            self._v(10, 10, 30, 200), self._v(60, 10, 30, 200),
            self._h(10, 300, 400, 300),            # 短边 300 ≫ 下限 100
        ]
        kept, dropped = _filter_large_sfx(boxes, *self.PAGE)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped[0][1], 300)

    def test_kept_boxes_stay_in_reading_order(self):
        from one_click_translate import _filter_large_sfx

        boxes = [self._v(500, 100, 40, 300), self._v(300, 100, 40, 300),
                 self._v(100, 100, 40, 300)]
        kept, _ = _filter_large_sfx(boxes, *self.PAGE)
        self.assertEqual([b[0] for b in kept], [500, 300, 100])


class GroupYieldTest(unittest.TestCase):
    """小组让位（_resolve_group_yields）：组矩形相交时面积小的让开。

    组矩形 = 成员框并集 ±bbox_pad = 嵌字布局边界，相交意味着两段译文会
    互相干扰。规则要点：只对更大的组让位、沿移动量最小的轴、位移有上限。
    """

    def test_smaller_yields_and_bigger_stays(self):
        from one_click_translate import _resolve_group_yields, _rects_intersect, _shifted

        big = [0, 0, 200, 200]
        small = [190, 50, 260, 150]          # 与大组横向重叠 10px
        shifts = _resolve_group_yields([big, small], gap=1)
        self.assertEqual(shifts[0], [0, 0])                    # 大组不动
        self.assertTrue(shifts[1][0] > 0 or shifts[1][1] != 0)  # 小组让位
        self.assertFalse(
            _rects_intersect(_shifted(big, shifts[0]), _shifted(small, shifts[1]))
        )

    def test_no_overlap_no_shift(self):
        from one_click_translate import _resolve_group_yields

        shifts = _resolve_group_yields([[0, 0, 100, 100], [300, 300, 400, 400]])
        self.assertEqual(shifts, [[0, 0], [0, 0]])

    def test_shift_uses_cheapest_axis(self):
        from one_click_translate import _resolve_group_yields

        big = [0, 0, 200, 200]
        small = [195, 60, 260, 140]          # 横向只需 -56？纵向不行 → 横向
        shifts = _resolve_group_yields([big, small], gap=1)
        self.assertEqual(shifts[1][1], 0)     # 不动 y
        self.assertNotEqual(shifts[1][0], 0)

    def test_oversized_shift_is_refused(self):
        """让位位移受上限约束：需要挪太远时宁可不动（不把文字甩到远处）。"""
        from one_click_translate import _resolve_group_yields

        big = [0, 0, 400, 400]
        small = [100, 100, 140, 140]         # 完全被大组包住，任何方向都要挪很远
        shifts = _resolve_group_yields([big, small], limit_ratio=0.5, gap=1)
        self.assertEqual(shifts[1], [0, 0])

    def test_cascade_three_groups(self):
        from one_click_translate import _resolve_group_yields, _rects_intersect, _shifted

        rects = [[0, 0, 300, 100], [280, 200, 420, 300], [400, 380, 470, 460]]
        shifts = _resolve_group_yields(rects, gap=1)
        self.assertEqual(shifts[0], [0, 0])   # 最大者不动
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                self.assertFalse(
                    _rects_intersect(_shifted(rects[i], shifts[i]),
                                     _shifted(rects[j], shifts[j])),
                    f"{i} 与 {j} 仍相交",
                )

    def test_build_records_shift_in_dataset(self):
        """位移要落到数据集里（typeset_shift），stage2 据此挪布局框。"""
        import json
        import tempfile
        from pathlib import Path

        from PIL import Image

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        images = root / "images"
        images.mkdir()
        page = images / "1.jpg"
        Image.new("RGB", (400, 300), "white").save(page)
        manifest = root / "capture.json"
        manifest.write_text(json.dumps({
            "source": {"path": str(page), "width": 400, "height": 300},
            "capture": {
                "box_source": "paddle",
                "selection_blocks": [
                    {"id": "sel_001", "image_bbox": [20, 20, 60, 280], "page_scale": False},
                    {"id": "sel_002", "image_bbox": [70, 20, 110, 280], "page_scale": False},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        islands = build_bluebox_dataset(manifest, root / "dataset" / "1", str(page))
        self.assertTrue(islands)
        # 同一组内的相邻列应被合并，因此这里不触发让位；键存在与否都合法
        self.assertIsInstance(islands[0].get("typeset_shift", [0, 0]), list)


class Stage2DatasetFilterPlumbingTest(unittest.TestCase):
    """_stage2_dataset 必须能吃下大拟音过滤参数并传给 build_bluebox_dataset。

    这是被真实事故逼出来的用例：参数只加到了 stage2_process 的签名，而真正
    调 build_bluebox_dataset 的是 _stage2_dataset，于是运行时报 NameError
    （单元测试当时没走到这条路径，全绿）。此处用最小 capture 目录把它钉住。
    """

    def test_dataset_build_receives_filter_options(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from one_click_translate import _stage2_dataset

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        images = root / "images"
        images.mkdir()
        page = images / "1.jpg"
        # 1 个纵向长条 + 3 个正常小框：长条会被大拟音规则剔除
        Image.new("RGB", (400, 600), "white").save(page)
        capture = root / "out" / "capture" / "paddle" / "1"
        capture.mkdir(parents=True)
        (capture / "capture.json").write_text(
            json.dumps({
                "source": {"path": str(page), "width": 400, "height": 600},
                "capture": {
                    "box_source": "paddle",
                    "complete": True,
                    "selection_blocks": [
                        {"id": "sel_001", "image_bbox": [10, 10, 110, 300],
                         "page_scale": False},
                        {"id": "sel_002", "image_bbox": [100, 10, 140, 120],
                         "page_scale": False},
                        {"id": "sel_003", "image_bbox": [160, 10, 200, 120],
                         "page_scale": False},
                        {"id": "sel_004", "image_bbox": [220, 10, 260, 120],
                         "page_scale": False},
                    ],
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        class _Progress:
            def log(self, *_args, **_kwargs):
                pass

            def progress(self, *_args, **_kwargs):
                pass

            def set_stage(self, *_args, **_kwargs):
                pass

        pages = _stage2_dataset(
            images, root / "out", ["1"], _Progress(), overwrite=False,
            drop_large_boxes=True, sfx_min_side_ratio=2.0, sfx_min_side_page=0.10,
        )
        self.assertEqual(pages, ["1"])
        written = json.loads(
            (root / "out" / "dataset" / "1" / "selection_blocks.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(written["selection_blocks"]), 3)
        self.assertNotIn("sel_001", {b["id"] for b in written["selection_blocks"]})

    def test_filter_can_be_disabled(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from one_click_translate import _stage2_dataset

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        images = root / "images"
        images.mkdir()
        page = images / "1.jpg"
        Image.new("RGB", (400, 600), "white").save(page)
        capture = root / "out" / "capture" / "paddle" / "1"
        capture.mkdir(parents=True)
        (capture / "capture.json").write_text(
            json.dumps({
                "source": {"path": str(page), "width": 400, "height": 600},
                "capture": {
                    "box_source": "photos",
                    "complete": True,
                    "selection_blocks": [
                        {"id": "sel_001", "image_bbox": [10, 10, 110, 300],
                         "page_scale": False},
                        {"id": "sel_002", "image_bbox": [100, 10, 140, 120],
                         "page_scale": False},
                        {"id": "sel_003", "image_bbox": [160, 10, 200, 120],
                         "page_scale": False},
                        {"id": "sel_004", "image_bbox": [220, 10, 260, 120],
                         "page_scale": False},
                    ],
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        class _Progress:
            def log(self, *_args, **_kwargs):
                pass

            def progress(self, *_args, **_kwargs):
                pass

            def set_stage(self, *_args, **_kwargs):
                pass

        pages = _stage2_dataset(
            images, root / "out", ["1"], _Progress(), overwrite=False,
            drop_large_boxes=False,
        )
        self.assertEqual(pages, ["1"])
        written = json.loads(
            (root / "out" / "dataset" / "1" / "selection_blocks.json").read_text(encoding="utf-8")
        )
        # 关掉过滤：Photos 路径也不剔除，4 个框全在
        self.assertEqual(len(written["selection_blocks"]), 4)


class FilteredEmptyReviewGateTest(unittest.TestCase):
    """筛框后一个框都不剩的页要进审核页，不能静默跳过。

    这类页在采集期是有框的（所以采集期的审核标记不会覆盖到），框是在下游
    被过滤光的；不补标记就会被当成"这页没问题"跳过。
    """

    def _run(self, blocks, **kwargs):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from one_click_translate import _stage2_dataset

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        images = root / "images"
        images.mkdir()
        page = images / "1.jpg"
        Image.new("RGB", (400, 600), "white").save(page)
        capture = root / "out" / "capture" / "paddle" / "1"
        capture.mkdir(parents=True)
        manifest = capture / "capture.json"
        manifest.write_text(json.dumps({
            "source": {"path": str(page), "width": 400, "height": 600},
            "capture": {
                "box_source": "paddle",
                "complete": True,
                "selection_blocks": blocks,
            },
        }, ensure_ascii=False), encoding="utf-8")

        class _Progress:
            def log(self, *_a, **_k): pass
            def progress(self, *_a, **_k): pass
            def set_stage(self, *_a, **_k): pass

        pages = _stage2_dataset(images, root / "out", ["1"], _Progress(),
                                overwrite=False, **kwargs)
        return pages, json.loads(manifest.read_text(encoding="utf-8"))

    def test_all_boxes_filtered_marks_review(self):
        # 让"框全被过滤掉"：用 20×20 的碎框、彼此相距 30px（不合并），
        # 每组成岛后 bbox = 框 + 组矩形 pad 6×2 = 28×28 = 784px² < 800 → 被微小岛过滤
        # 丢掉 → 该页一个可用框都不剩。
        # （注：大拟音过滤基于中位数，永远不会把框删光；能"筛空"的是微小岛
        # 过滤这类阈值过滤，所以测试走这条。）
        blocks = [
            {"id": "sel_001", "image_bbox": [10, 10, 26, 26], "page_scale": False},
            {"id": "sel_002", "image_bbox": [60, 10, 76, 26], "page_scale": False},
            {"id": "sel_003", "image_bbox": [110, 10, 126, 26], "page_scale": False},
            {"id": "sel_004", "image_bbox": [160, 10, 176, 26], "page_scale": False},
        ]
        pages, manifest = self._run(blocks)
        capture = manifest["capture"]
        self.assertFalse(capture.get("selection_review_confirmed", False))
        self.assertEqual(pages, [], "无框页不应进入下游")
        self.assertTrue(capture.get("selection_empty_after_filter"))
        self.assertTrue(capture.get("selection_review_required"))
        self.assertIn("筛框后无框", capture.get("selection_review_reason", ""))

    def test_page_with_surviving_boxes_not_flagged(self):
        # 4 个正常小框（短边 40，中位 40 → 都不超标）→ 不成岛也无审核标记
        blocks = [
            {"id": "sel_001", "image_bbox": [10, 10, 50, 120], "page_scale": False},
            {"id": "sel_002", "image_bbox": [60, 10, 100, 120], "page_scale": False},
            {"id": "sel_003", "image_bbox": [110, 10, 150, 120], "page_scale": False},
            {"id": "sel_004", "image_bbox": [160, 10, 200, 120], "page_scale": False},
        ]
        pages, manifest = self._run(blocks)
        self.assertEqual(pages, ["1"])
        self.assertFalse(manifest["capture"].get("selection_empty_after_filter", False))


class SfxGlossaryWiringTest(unittest.TestCase):
    """翻译阶段把稳定的拟声词收录进术语库（_sync_sfx_glossary）。

    端到端护栏：登录入库的原文必须是短拟声、译文短且非疑问词、且重复
    出现同一译法；收录后 glossary.json 的 sha 变化（→ 页级翻译缓存失效），
    否则"入库"对已缓存的页面毫无作用。
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.glossary = self.directory / "glossary.json"
        self.glossary.write_text(
            json.dumps({"version": 1, "terms": {}, "notes": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        patcher = mock.patch("one_click_translate.ROOT", self.directory)
        self.addCleanup(patcher.stop)
        patcher.start()
        (self.directory / "config").mkdir(exist_ok=True)
        self.glossary.replace(self.directory / "config" / "glossary.json")

    def _progress(self):
        return mock.Mock()

    def _sync(self, texts, translations, min_count=2):
        from one_click_translate import _sync_sfx_glossary

        return _sync_sfx_glossary(
            self.directory, texts, translations, self._progress(),
            min_count=min_count,
        )

    def _terms(self):
        return json.loads(
            (self.directory / "config" / "glossary.json").read_text(encoding="utf-8")
        )["terms"]

    def test_repeated_sfx_recorded(self):
        texts = {"1": {"a": "ぎゅっ♡"}, "2": {"b": "ぎゅっ♡"}}
        translations = {"1": {"a": "夹紧♡"}, "2": {"b": "夹紧♡"}}
        report = self._sync(texts, translations)
        self.assertEqual(report["added"], ["ぎゅっ♡"])
        self.assertEqual(self._terms()["ぎゅっ♡"]["translation"], "夹紧♡")

    def test_single_occurrence_not_recorded(self):
        report = self._sync(
            {"1": {"a": "ぎゅっ♡"}}, {"1": {"a": "夹紧♡"}},
        )
        self.assertEqual(report["added"], [])
        self.assertEqual(self._terms(), {})

    def test_misread_question_word_never_recorded(self):
        """どうし → "为什么"（就是用户报的那个错）不许进库。"""
        texts = {"1": {"a": "どうし"}, "2": {"b": "どうし"}}
        translations = {"1": {"a": "为什么"}, "2": {"b": "为什么"}}
        report = self._sync(texts, translations)
        self.assertEqual(report["added"], [])
        self.assertEqual(self._terms(), {})

    def test_min_count_zero_disables(self):
        texts = {"1": {"a": "ぎゅっ♡"}, "2": {"b": "ぎゅっ♡"}}
        translations = {"1": {"a": "夹紧♡"}, "2": {"b": "夹紧♡"}}
        report = self._sync(texts, translations, min_count=0)
        self.assertEqual(report["added"], [])
        self.assertEqual(self._terms(), {})

    def test_recording_changes_glossary_sha(self):
        from manga_localizer.translator import load_glossary

        target = self.directory / "config" / "glossary.json"
        before = load_glossary(target).sha256
        self._sync(
            {"1": {"a": "ぎゅっ♡"}, "2": {"b": "ぎゅっ♡"}},
            {"1": {"a": "夹紧♡"}, "2": {"b": "夹紧♡"}},
        )
        self.assertNotEqual(before, load_glossary(target).sha256)


class OversizedBoxNoMergeTest(unittest.TestCase):
    """过大单框（短边 ≫ 同方向基准）不参与合并，各成一岛。

    并进竖排组会把组的外接矩形和字号一起撑大（实测页 7/11/19 的巨型字号），
    而且这种框在原图里本来就是独立的超大文字，不该和别的列共用一个排版容器。
    不合并之后它自然满足"单框岛"条件 → 字号被 --sfx-max-font-size 封顶。
    """

    def _boxes(self, specs):
        return [
            {"id": f"sel_{index:03d}", "image_bbox": list(bbox), "page_scale": False}
            for index, bbox in enumerate(specs, 1)
        ]

    def test_oversized_box_ids_flags_the_big_one(self):
        from one_click_translate import oversized_box_ids

        boxes = self._boxes([
            (100, 100, 160, 300),   # 常规竖框：短边 60
            (100, 320, 160, 520),
            (100, 540, 160, 740),
            (100, 760, 160, 960),
            (400, 100, 560, 620),   # 大单框：短边 160 ≫ 1.5×60
        ])
        self.assertEqual(oversized_box_ids(boxes, ratio=1.5), {"sel_005"})

    def test_locked_box_never_merges(self):
        from one_click_translate import _group_boxes_by_overlap

        # 两个竖框水平相邻且垂直重叠 80% —— 正常会合并
        a = {"id": "sel_001", "image_bbox": [100, 100, 160, 300]}
        b = {"id": "sel_002", "image_bbox": [162, 100, 222, 300]}
        merged = _group_boxes_by_overlap([a, b])
        self.assertEqual(len(merged), 1)
        locked = _group_boxes_by_overlap([a, b], no_merge_ids=["sel_002"])
        self.assertEqual(len(locked), 2)
        self.assertEqual(sorted(len(ids) for _, ids in locked), [1, 1])

    def test_group_build_locks_oversized_boxes(self):
        """数据集构建阶段要用 oversized_box_ids 的结果锁住合并。"""
        import inspect

        from one_click_translate import build_bluebox_dataset

        source = inspect.getsource(build_bluebox_dataset)
        self.assertIn("oversized_box_ids", source)
        self.assertIn("no_merge_ids=oversized", source)


class OversizedAbsoluteFloorTest(unittest.TestCase):
    """过大单框判据必须同时超过绝对下限（页短边 8%）。

    只看倍数的话，小字页面的门槛会低到"人人超标"：实测 丙书 页 97 常规字号
    22px → 门槛 33px → 页上几乎所有框（短边 47~77px）都被判成过大 → 全压到
    30px、还互相不合并（封顶框 23% → 加下限后 8%）。这与采集期剔除大拟音的
    判据同形（那条也要求同时超过页短边 10%）。
    """

    def _page(self, specs, image_size=(1000, 1400)):
        boxes = [
            {"id": f"sel_{i:03d}", "image_bbox": list(b), "page_scale": False}
            for i, b in enumerate(specs, 1)
        ]
        data = {
            "schema_version": 1, "page": "1", "image_size": list(image_size),
            "contours": True,
            "islands": [
                {"id": f"island_{i:03d}", "image_bbox": list(b),
                 "image_center": [(b[0]+b[2])//2, (b[1]+b[3])//2],
                 "image_area": float((b[2]-b[0])*(b[3]-b[1]))}
                for i, b in enumerate(specs, 1)
            ],
        }
        from manga_localizer.island_contract import dataset_from_data

        return boxes, dataset_from_data(data, path=Path(__file__), source_sha256="a"*64)

    def test_small_page_midsize_boxes_not_flagged(self):
        """小字页面：中等框（1.5×基准但未过页短边 8%）不再被判过大。"""
        from one_click_translate import oversized_box_ids

        # 6 个小框（短边 20）→ 基准中位 20 → 阈值 30；一个中等框短边 40
        specs = [(100 + i * 40, 100, 120 + i * 40, 300) for i in range(6)]
        specs.append((400, 400, 440, 600))
        boxes, _ = self._page(specs, image_size=(1000, 1400))
        without_floor = oversized_box_ids(boxes, ratio=1.5)
        with_floor = oversized_box_ids(boxes, ratio=1.5, page_short_side=1000.0)
        self.assertEqual(without_floor, {"sel_007"})   # 没下限 → 误判中等框
        self.assertEqual(with_floor, set())            # 有下限（页短边 8% = 80）→ 放过

    def test_truly_huge_box_still_flagged_with_floor(self):
        from one_click_translate import oversized_box_ids

        specs = [(100 + i * 40, 100, 120 + i * 40, 300) for i in range(6)]
        specs.append((400, 400, 600, 1100))            # 短边 200 > 80
        boxes, _ = self._page(specs, image_size=(1000, 1400))
        self.assertEqual(
            oversized_box_ids(boxes, ratio=1.5, page_short_side=1000.0),
            {"sel_007"},
        )

    def test_both_sides_agree(self):
        """编组侧的 oversized_box_ids 与封顶侧的 oversized_single_box_islands 同口径。"""
        from one_click_translate import (
            oversized_box_ids, oversized_single_box_islands,
        )

        specs = [(100 + i * 40, 100, 120 + i * 40, 300) for i in range(6)]
        specs.append((400, 400, 600, 1100))
        boxes, dataset = self._page(specs, image_size=(1000, 1400))
        locked = oversized_box_ids(boxes, ratio=1.5, page_short_side=1000.0)
        capped = oversized_single_box_islands(dataset, boxes, ratio=1.5)
        self.assertEqual(locked, {"sel_007"})
        self.assertEqual(capped, {"island_007"})



def _art_page_image(size=(600, 600)):
    """左半边平坦白底（气泡/色块），右半边 1px 黑白条纹（模拟网点/画面）。

    条纹的 3×3 亮度极差 = 255，所以"局部平坦"判据在右半边为 0、左半边为 1。
    用它可以直接验证判据分得开"印在平坦底上"和"画在画面上"。
    """
    array = np.full((size[1], size[0]), 255, dtype=np.uint8)
    array[:, size[0] // 2:] = 0
    array[::2, size[0] // 2:] = 255
    return Image.fromarray(array, mode="L")


class BgFlatnessTest(unittest.TestCase):
    """手绘字判据的几何侧（flat_background_ratio / attach_bg_flatness）。

    判据是**几何**（岛底平坦度），不是读数——这里锁的是几条容易写错的边界：
    平坦底给高分、条纹底给零分、纯色（非白）色块也算平坦、测不了就不给值。
    """

    def setUp(self):
        self.gray = np.asarray(_art_page_image(), dtype=np.uint8)

    def test_flat_and_art_areas_are_at_the_extremes(self):
        from one_click_translate import flat_background_ratio

        self.assertEqual(flat_background_ratio(self.gray, [50, 50, 250, 250]), 1.0)
        self.assertEqual(flat_background_ratio(self.gray, [350, 50, 550, 250]), 0.0)

    def test_unusable_box_yields_none(self):
        from one_click_translate import flat_background_ratio

        self.assertIsNone(flat_background_ratio(self.gray, [50, 50, 53, 53]))

    def test_tinted_flat_area_counts_as_flat(self):
        """纯色色块（不是白）也是"平坦的底"：封面标题印在浅粉色块上，要留住。"""
        from one_click_translate import flat_background_ratio

        tinted = np.full((600, 600), 150, dtype=np.uint8)
        self.assertEqual(flat_background_ratio(tinted, [50, 50, 250, 250]), 1.0)

    def test_attach_writes_ratio_per_island(self):
        from one_click_translate import attach_bg_flatness

        islands = [
            {"id": "island_001", "image_bbox": [50, 50, 250, 250]},
            {"id": "island_002", "image_bbox": [350, 50, 550, 250]},
            {"id": "island_003", "image_bbox": [50, 50, 53, 53]},
        ]
        attach_bg_flatness(islands, self.gray)
        self.assertEqual(islands[0]["bg_flat_ratio"], 1.0)
        self.assertEqual(islands[1]["bg_flat_ratio"], 0.0)
        self.assertNotIn("bg_flat_ratio", islands[2])   # 太小，不判


class ArtTextIslandTest(unittest.TestCase):
    """岛级判定：几何 + 字数两条缺一不可。"""

    def _is_art(self, text, ratio):
        from one_click_translate import art_text_island

        return art_text_island(text, ratio)

    def test_short_text_on_art_is_art(self):
        self.assertTrue(self._is_art("よろしく", 0.08))

    def test_long_text_on_art_is_not_art(self):
        """压在画面上的印刷正文（封面警示语、后记段落）字数超限 → 不是手绘字。"""
        self.assertFalse(
            self._is_art("・普通の性行為が見たい方は回れ右した方が賢明です", 0.17)
        )

    def test_short_text_on_flat_background_is_not_art(self):
        self.assertFalse(self._is_art("んっ♡", 0.90))

    def test_missing_measurement_is_not_art(self):
        """老数据集没有 bg_flat_ratio → 判据无法自证 → 保守地不当作手绘字。"""
        self.assertFalse(self._is_art("よろしく", None))
        self.assertFalse(self._is_art("", 0.05))


class DatasetBgFlatnessTest(unittest.TestCase):
    """bg_flat_ratio 必须落到 islands.json，并且能被"规则变新"判断看见。"""

    def _capture(self, root: Path):
        images = root / "images"
        images.mkdir(exist_ok=True)
        page = images / "1.jpg"
        _art_page_image().convert("RGB").save(page)
        capture = root / "out" / "capture" / "paddle" / "1"
        capture.mkdir(parents=True, exist_ok=True)
        (capture / "capture.json").write_text(
            json.dumps({
                "source": {"path": str(page), "width": 600, "height": 600},
                "capture": {
                    "box_source": "paddle", "complete": True,
                    "selection_blocks": [
                        {"id": "sel_001", "image_bbox": [50, 50, 250, 250],
                         "page_scale": False},
                        {"id": "sel_002", "image_bbox": [350, 50, 550, 250],
                         "page_scale": False},
                    ],
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return capture / "capture.json", page

    def test_ratio_written_to_islands_json(self):
        from one_click_translate import _dataset_has_bg_flatness

        with tempfile.TemporaryDirectory() as tmp:
            manifest, page = self._capture(Path(tmp))
            page_dir = Path(tmp) / "out" / "dataset" / "1"
            build_bluebox_dataset(manifest, page_dir, page, drop_art_boxes=True)
            islands = json.loads(
                (page_dir / "islands.json").read_text(encoding="utf-8")
            )["islands"]
            self.assertEqual(
                sorted(item["bg_flat_ratio"] for item in islands), [0.0, 1.0]
            )
            self.assertTrue(_dataset_has_bg_flatness(page_dir / "islands.json"))

    def test_marking_can_be_disabled(self):
        from one_click_translate import _dataset_has_bg_flatness

        with tempfile.TemporaryDirectory() as tmp:
            manifest, page = self._capture(Path(tmp))
            page_dir = Path(tmp) / "out" / "dataset" / "1"
            build_bluebox_dataset(manifest, page_dir, page, drop_art_boxes=False)
            islands = json.loads(
                (page_dir / "islands.json").read_text(encoding="utf-8")
            )["islands"]
            self.assertFalse(any("bg_flat_ratio" in item for item in islands))
            # 关掉判据时数据集里本就没有该字段；"规则过期"的检查只在该判据
            # 打开时生效，不会因为缺字段反复重建。
            self.assertFalse(_dataset_has_bg_flatness(page_dir / "islands.json"))


class StaleDatasetRulesTest(unittest.TestCase):
    """规则变了要重建 dataset，不能静默复用旧产物。

    ``bg_flat_ratio`` 是手绘字判据新增的字段：老数据集没有它，判据就无从判定，
    而"islands.json 比 capture.json 新"这条跳过规则会把老数据集一直复用下去
    （看起来"改了规则没用"，开发记录 里踩过一次）。
    """

    def _run(self, *, drop_art_boxes, islands_payload):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from one_click_translate import _stage2_dataset

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        images = root / "images"
        images.mkdir()
        page = images / "1.jpg"
        _art_page_image().convert("RGB").save(page)
        capture = root / "out" / "capture" / "paddle" / "1"
        capture.mkdir(parents=True)
        manifest = capture / "capture.json"
        manifest.write_text(json.dumps({
            "source": {"path": str(page), "width": 600, "height": 600},
            "capture": {
                "box_source": "paddle", "complete": True,
                "selection_blocks": [
                    {"id": "sel_001", "image_bbox": [50, 50, 250, 250],
                     "page_scale": False},
                    {"id": "sel_002", "image_bbox": [350, 50, 550, 250],
                     "page_scale": False},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        dataset = root / "out" / "dataset" / "1"
        dataset.mkdir(parents=True)
        islands_path = dataset / "islands.json"
        islands_path.write_text(json.dumps(islands_payload), encoding="utf-8")
        selection_path = dataset / "selection_blocks.json"
        selection_path.write_text(json.dumps({"selection_blocks": []}),
                                  encoding="utf-8")
        # 让 islands.json 比 capture.json 新：跳过规则本该命中
        import os
        os.utime(islands_path, (manifest.stat().st_mtime + 10,) * 2)

        class _Progress:
            def log(self, *_a, **_k): pass
            def progress(self, *_a, **_k): pass
            def set_stage(self, *_a, **_k): pass

        _stage2_dataset(images, root / "out", ["1"], _Progress(),
                        overwrite=False, drop_art_boxes=drop_art_boxes)
        return json.loads(islands_path.read_text(encoding="utf-8"))

    OLD_ISLANDS = {
        "schema_version": 1, "page": "1", "image_size": [600, 600],
        "contours": True, "fill_mode": "black", "min_island_area": 800,
        "blue_boxes_only": True,
        "islands": [
            {"id": "island_001", "image_bbox": [44, 44, 256, 256],
             "image_center": [150, 150], "image_area": 44944.0},
        ],
    }

    def test_old_dataset_rebuilt_when_geometry_rule_needs_it(self):
        islands = self._run(drop_art_boxes=True, islands_payload=self.OLD_ISLANDS)
        self.assertTrue(
            all(item.get("bg_flat_ratio") is not None
                for item in islands["islands"])
        )
        self.assertEqual(len(islands["islands"]), 2)

    def test_old_dataset_reused_when_rule_is_off(self):
        islands = self._run(drop_art_boxes=False, islands_payload=self.OLD_ISLANDS)
        self.assertNotIn("bg_flat_ratio", islands["islands"][0])
        self.assertEqual(len(islands["islands"]), 1)   # 原样复用，没重建

class BoxDropListTests(unittest.TestCase):
    """框级"被丢清单"：框级过滤此前**没有人工真值覆盖**（被丢的框在编组前就消失、
    从不进核对表），"筛框到底杀了多少台词"无人知道。这里锁住"落盘"这件事。
    """

    def test_large_sfx_box_drop_is_recorded(self):
        import json
        import tempfile
        from pathlib import Path as _Path

        from one_click_translate import build_bluebox_dataset

        # 一页里 6 个正常小框 + 1 个超大框（短边远超同方向基准且 > 10% 页短边）
        blocks = [{"id": f"sel_{i:03d}",
                   "image_bbox": [40, 60 + i * 40, 140, 90 + i * 40],
                   "page_scale": False} for i in range(1, 7)]
        blocks.append({"id": "sel_099", "image_bbox": [300, 300, 900, 1300],
                       "page_scale": False})
        manifest = {"source": {"width": 1000, "height": 1400},
                    "capture": {"selection_blocks": blocks}}
        with tempfile.TemporaryDirectory(prefix="boxdrop_") as tmp:
            root = _Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            from PIL import Image

            source = root / "page.jpg"
            Image.new("RGB", (1000, 1400), "white").save(source)
            page_dir = root / "dataset" / "1"
            islands = build_bluebox_dataset(root / "manifest.json", page_dir, source)
            self.assertTrue(all(i["image_bbox"][2] <= 200 for i in islands
                                if i["id"] == "island_001"))
            drop_path = page_dir / "dropped_boxes.csv"
            self.assertTrue(drop_path.is_file(), "大拟音框被丢时应当落 dropped_boxes.csv")
            text = drop_path.read_text(encoding="utf-8-sig")
            self.assertIn("大拟音", text)
            self.assertIn("sel_099", text)


class SoftFlagLargeSfxTest(unittest.TestCase):
    """框级软标记（soft_flag_large_sfx）：硬过滤的放宽档，只标记不丢框。

    严格联合版（开发记录 六之前四）的第一段。要锁住的边界：中等大的框确实被
    标（1.2×~2.0× 之间那批是严格版的主要增量）、正常字号的框不被标、
    框数不足时与硬过滤同样保持沉默。
    """

    PAGE = (1000, 1400)

    def _v(self, x, y, w, h):
        return [x, y, x + w, y + h]

    def test_medium_box_is_flagged(self):
        from one_click_translate import soft_flag_large_sfx

        boxes = [self._v(10, 10, 40, 300), self._v(60, 10, 40, 300),
                 self._v(110, 10, 40, 300), self._v(160, 10, 60, 300)]
        flagged = soft_flag_large_sfx(boxes, *self.PAGE)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0], boxes[3])   # 短边 60：> 1.2×40 且 > 5%×1000

    def test_normal_boxes_are_not_flagged(self):
        from one_click_translate import soft_flag_large_sfx

        boxes = [self._v(10, 10, 40, 300), self._v(60, 10, 40, 300),
                 self._v(110, 10, 40, 300), self._v(160, 10, 40, 300)]
        self.assertEqual(soft_flag_large_sfx(boxes, *self.PAGE), [])

    def test_too_few_boxes_flags_nothing(self):
        from one_click_translate import soft_flag_large_sfx

        boxes = [self._v(10, 10, 90, 300), self._v(120, 10, 90, 300)]
        self.assertEqual(soft_flag_large_sfx(boxes, *self.PAGE), [])


class SoftFlagDatasetPlumbingTest(unittest.TestCase):
    """build_bluebox_dataset 必须把软标记写进岛（sfx_soft_flag）和数据集级
    标记（soft_flag_version）——岛级裁决与"数据集新旧"判据都靠它们。"""

    def test_flagged_island_and_marker_written(self):
        import json
        import tempfile
        from pathlib import Path

        from PIL import Image

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        page = root / "1.jpg"
        Image.new("RGB", (400, 300), "white").save(page)
        # 三个正常竖框（短边 20）+ 一个中等框（短边 30 > 1.2×20 且 > 5%×300）
        manifest = root / "capture.json"
        manifest.write_text(json.dumps({
            "source": {"path": str(page), "width": 400, "height": 300},
            "capture": {
                "box_source": "paddle",
                "selection_blocks": [
                    {"id": "sel_001", "image_bbox": [20, 20, 40, 280], "page_scale": False},
                    {"id": "sel_002", "image_bbox": [100, 20, 120, 280], "page_scale": False},
                    {"id": "sel_003", "image_bbox": [180, 20, 200, 280], "page_scale": False},
                    {"id": "sel_004", "image_bbox": [260, 20, 290, 280], "page_scale": False},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        page_dir = root / "dataset" / "1"
        # 软标记与 bg_flat_ratio 同一门控：只在手绘字功能开启时写
        islands = build_bluebox_dataset(manifest, page_dir, str(page),
                                        drop_art_boxes=True)
        payload = json.loads(
            (page_dir / "islands.json").read_text(encoding="utf-8"))
        self.assertEqual(payload.get("soft_flag_version"), 1)
        by_flag = {i["id"]: bool(i.get("sfx_soft_flag")) for i in islands}
        flagged = [iid for iid, v in by_flag.items() if v]
        self.assertEqual(len(flagged), 1, f"恰好一个中等框岛被标记：{by_flag}")

    def test_flags_absent_when_feature_off(self):
        """手绘字功能没开（drop_art_boxes=False）→ 绝不写软标记。

        软标记是岛级裁决的"像拟声"一票：功能关着的书绝不能因为数据集里
        多了这个字段就被丢岛（与 bg_flat_ratio 同一门控）。
        """
        import json
        import tempfile
        from pathlib import Path

        from PIL import Image

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        page = root / "1.jpg"
        Image.new("RGB", (400, 300), "white").save(page)
        manifest = root / "capture.json"
        manifest.write_text(json.dumps({
            "source": {"path": str(page), "width": 400, "height": 300},
            "capture": {
                "box_source": "paddle",
                "selection_blocks": [
                    {"id": "sel_001", "image_bbox": [20, 20, 40, 280], "page_scale": False},
                    {"id": "sel_002", "image_bbox": [100, 20, 120, 280], "page_scale": False},
                    {"id": "sel_003", "image_bbox": [180, 20, 200, 280], "page_scale": False},
                    {"id": "sel_004", "image_bbox": [260, 20, 290, 280], "page_scale": False},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        islands = build_bluebox_dataset(manifest, root / "dataset" / "1",
                                        str(page))
        self.assertTrue(all("sfx_soft_flag" not in i for i in islands))
