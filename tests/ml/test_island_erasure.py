from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageChops, ImageDraw

from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_erasure import erase_island_image
from manga_localizer.eraser import (
    ErasureConfig, LamaInpainter, OpenCVFSRInpainter, select_inpainter,
)


def _dataset(islands: list[dict], size=(200, 300)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


def _white_page_with_black_text(size=(200, 300)):
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(15, 55, 5):
        draw.line((x, 30, x, 80), fill=(0, 0, 0), width=2)
    return image


class IslandErasureTest(unittest.TestCase):
    def test_polygon_island_erase_keeps_outside_untouched(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                    "image_polygon": [[10, 20], [60, 20], [60, 90], [10, 90]],
                }
            ]
        )
        image = _white_page_with_black_text()
        result = erase_island_image(image, dataset, config=ErasureConfig())
        # hard pixel-integrity guarantee
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertTrue(result.pixels.outside_mask_unchanged)
        self.assertEqual(result.island_pixels.changed_outside_mask_pixels, 0)
        # glyph pixels inside the island were erased
        self.assertGreater(result.island_pixels.changed_inside_mask_pixels, 0)
        self.assertEqual(len(result.columns), 1)
        self.assertEqual(result.columns[0].column_id, "island_001")
        self.assertEqual(result.columns[0].strategy, "solid_fill")
        # erased output differs from source only inside the island
        difference = result.difference
        self.assertEqual(difference.getpixel((5, 5)), (0, 0, 0))
        self.assertNotEqual(difference.getpixel((20, 40)), (0, 0, 0))

    def test_bbox_fallback_island(self):
        dataset = _dataset(
            [
                {
                    "id": "island_002",
                    "image_bbox": [100, 30, 150, 120],
                    "image_center": [125, 75],
                    "image_area": 1.0,
                }
            ]
        )
        image = _white_page_with_black_text()
        result = erase_island_image(image, dataset, config=ErasureConfig())
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertEqual(result.columns[0].column_id, "island_002")
        # island has no ink, so nothing inside changes either
        self.assertEqual(result.island_pixels.changed_inside_mask_pixels, 0)

    def test_forced_inpaint_strategy(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 20, 60, 90],
                    "image_center": [35, 55],
                    "image_area": 1.0,
                    "image_polygon": [[10, 20], [60, 20], [60, 90], [10, 90]],
                }
            ]
        )
        image = _white_page_with_black_text()
        result = erase_island_image(
            image, dataset, config=ErasureConfig(style_override="inpaint")
        )
        self.assertTrue(result.island_pixels.outside_mask_unchanged)
        self.assertEqual(result.columns[0].strategy, "inpaint")
        self.assertEqual(result.columns[0].classifier, "cli-override")


class _FakeLamaSession:
    """假 LaMa 会话：不管输入，返回整片纯色——用来验证**像素契约**而不是模型质量。"""

    def __init__(self, color=(0, 255, 0)):
        import numpy as np

        self._output = np.zeros((1, 3, 512, 512), dtype=np.float32)
        for channel, value in enumerate(color):
            self._output[:, channel] = value / 255.0

    def run(self, output_names, input_feed):
        return [self._output]


class InpainterSelectionTest(unittest.TestCase):
    """修复后端的挑选：显式 fsr/lama + auto 的可用性回退。"""

    def test_explicit_fsr(self):
        self.assertIsInstance(
            select_inpainter(ErasureConfig(inpainter="fsr")), OpenCVFSRInpainter)

    def test_explicit_lama_without_model_raises(self):
        from manga_localizer.errors import ErasureError

        missing = LamaInpainter(model_path=Path("nonexistent/model.onnx"))
        with mock.patch("manga_localizer.eraser.LamaInpainter",
                        return_value=missing):
            with self.assertRaises(ErasureError):
                select_inpainter(ErasureConfig(inpainter="lama"))

    def test_auto_falls_back_to_fsr_without_model(self):
        missing = LamaInpainter(model_path=Path("nonexistent/model.onnx"))
        with mock.patch("manga_localizer.eraser.LamaInpainter",
                        return_value=missing):
            self.assertIsInstance(select_inpainter(ErasureConfig()),
                                  OpenCVFSRInpainter)

    def test_auto_prefers_lama_when_available(self):
        with mock.patch.object(LamaInpainter, "available", return_value=True):
            self.assertIsInstance(select_inpainter(ErasureConfig()), LamaInpainter)

    def test_available_is_false_for_missing_model(self):
        self.assertFalse(LamaInpainter(model_path=Path("nope.onnx")).available())


class LamaInpainterContractTest(unittest.TestCase):
    """LaMa 修复必须守住擦字的**像素完整性契约**：掩膜外逐位不变。"""

    def _mask_and_image(self, size=(400, 300)):
        image = Image.new("RGB", size, (200, 200, 200))
        ImageDraw.Draw(image).rectangle([150, 100, 250, 180], fill=(20, 20, 20))
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle([140, 90, 260, 190], fill=255)
        return image, mask

    def _inpaint(self, image, mask):
        inpainter = LamaInpainter(model_path=Path("fake.onnx"))
        object.__setattr__(inpainter, "_cached_session", _FakeLamaSession())
        return inpainter.inpaint(image, mask, 8)

    def test_pixels_outside_the_mask_are_bit_identical(self):
        image, mask = self._mask_and_image()
        out = self._inpaint(image, mask)
        changed = ImageChops.difference(image, out.convert("RGB")).convert("L").point(
            lambda value: 255 if value > 0 else 0)
        leak = ImageChops.subtract(changed, mask)
        self.assertIsNone(leak.getbbox(), "掩膜外的像素被改动了")

    def test_pixels_inside_the_mask_do_change(self):
        """否则"契约通过"是空的——必须确认它真的填了。"""
        image, mask = self._mask_and_image()
        out = self._inpaint(image, mask)
        inside = ImageChops.multiply(
            ImageChops.difference(image, out.convert("RGB")).convert("L").point(
                lambda value: 255 if value > 0 else 0),
            mask)
        self.assertIsNotNone(inside.getbbox())

    def test_empty_mask_returns_a_copy(self):
        image = Image.new("RGB", (64, 64), (10, 20, 30))
        out = self._inpaint(image, Image.new("L", (64, 64), 0))
        self.assertIsNone(ImageChops.difference(image, out).getbbox())

    def test_tiny_component_is_skipped(self):
        """小于下限的连通域不跑模型（也不该被改）。"""
        image = Image.new("RGB", (64, 64), (10, 20, 30))
        mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(mask).point((30, 30), fill=255)
        out = self._inpaint(image, mask)
        self.assertIsNone(ImageChops.difference(image, out).getbbox())


class EraseCacheInpainterKeyTest(unittest.TestCase):
    """擦字缓存键必须含修复后端——否则换 LaMa 会串用 FSR 的旧结果。"""

    def test_inpainter_choice_changes_the_key(self):
        from one_click_translate import _erase_cache_key

        regions = {"island_001": [0, 0, 10, 10]}
        fsr = _erase_cache_key("a" * 64, regions, 800.0,
                              ErasureConfig(inpainter="fsr"))
        lama = _erase_cache_key("a" * 64, regions, 800.0,
                               ErasureConfig(inpainter="lama"))
        auto = _erase_cache_key("a" * 64, regions, 800.0, ErasureConfig())
        self.assertNotEqual(fsr, lama)
        self.assertNotEqual(fsr, auto)


if __name__ == "__main__":
    unittest.main()
