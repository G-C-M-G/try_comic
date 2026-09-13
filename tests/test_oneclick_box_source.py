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
