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
            (self.out_root / "capture" / "7" / "capture.json").read_text(encoding="utf-8")
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
        manifest_path = self.out_root / "capture" / "7" / "capture.json"
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
            {"polygon": _poly(260, 60, 300, 300), "text": "い", "confidence": 0.9},
        ])
        groups = manifest["capture"]["selection_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["members"]), 2)

    def test_group_helper_accepts_generated_blocks(self):
        blocks = [
            {"id": "sel_001", "image_bbox": [200, 60, 240, 300], "page_scale": False},
            {"id": "sel_002", "image_bbox": [260, 60, 300, 300], "page_scale": False},
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
