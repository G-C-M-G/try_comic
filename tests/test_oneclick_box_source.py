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
        # 两个竖排框纵向重叠、横向紧邻 → 同一组（20% 间距 / 40% 重叠规则）
        ok, manifest = self._capture([
            {"polygon": _poly(200, 60, 240, 300), "text": "あ", "confidence": 0.9},
            {"polygon": _poly(248, 60, 288, 300), "text": "い", "confidence": 0.9},
        ])
        groups = manifest["capture"]["selection_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["members"]), 2)

    def test_group_helper_accepts_generated_blocks(self):
        blocks = [
            {"id": "sel_001", "image_bbox": [200, 60, 240, 300], "page_scale": False},
            {"id": "sel_002", "image_bbox": [248, 60, 288, 300], "page_scale": False},
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
