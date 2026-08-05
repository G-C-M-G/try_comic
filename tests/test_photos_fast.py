import unittest

import numpy as np

from photos_capture import (
    OverlayBlock,
    _assign_paddle_predictions,
    _build_island_mask,
    _display_bbox_to_image,
    _finalize_paddle_column,
    _image_bbox_to_display,
    _split_column_bands,
    _text_acceptable,
)


class IslandMaskTests(unittest.TestCase):
    def test_blackens_outside_islands_and_keeps_inside(self):
        image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3) % 255
        masked = _build_island_mask(image, [[10, 10, 40, 50]])

        self.assertTrue(np.all(masked[20:40, 20:35] == image[20:40, 20:35]))
        self.assertTrue(np.all(masked[5:8, 5:8] == 0))
        self.assertTrue(np.all(masked[55:58, 70:75] == 0))


class ColumnBandTests(unittest.TestCase):
    def _island_with_columns(self, height=120, width=200):
        active = np.full((160, 240, 3), 230, dtype=np.uint8)
        left = slice(30, 50)
        right = slice(130, 150)
        rows = slice(20, 100)
        active[rows, left] = 30
        active[rows, right] = 30
        return active

    def test_splits_island_into_two_vertical_bands(self):
        active = self._island_with_columns()
        bands = _split_column_bands(active, OverlayBlock(20, 10, 170, 110))

        self.assertEqual(len(bands), 2)
        left, right = sorted(bands, key=lambda band: band[0])
        self.assertLess(left[0], 35)
        self.assertGreaterEqual(left[2], 45)
        self.assertLess(right[0], 135)
        self.assertGreaterEqual(right[2], 145)

    def test_returns_whole_block_when_no_ink(self):
        active = np.full((100, 140, 3), 240, dtype=np.uint8)
        block = OverlayBlock(10, 10, 60, 40)
        self.assertEqual(_split_column_bands(active, block), [[10, 10, 70, 50]])

    def test_band_height_follows_ink_extent(self):
        active = np.full((160, 240, 3), 230, dtype=np.uint8)
        active[60:90, 30:45] = 30
        bands = _split_column_bands(active, OverlayBlock(20, 10, 50, 120))

        self.assertEqual(len(bands), 1)
        self.assertLessEqual(bands[0][1], 62)
        self.assertGreaterEqual(bands[0][3], 88)


class PaddleAssignmentTests(unittest.TestCase):
    def _column(self, image_bbox, direction="vertical"):
        return {
            "id": "column_001",
            "text": "",
            "paddle_text": "",
            "paddle_confidence": None,
            "text_source": "",
            "needs_review": False,
            "image_bbox": image_bbox,
            "direction": direction,
            "overlay_block": 0,
        }

    def test_joins_vertical_hits_top_to_bottom(self):
        column = self._column([100, 50, 120, 150])
        predictions = [
            {"polygon": [[115, 120], [120, 120], [120, 140], [115, 140]], "text": "下", "confidence": 0.95},
            {"polygon": [[115, 60], [120, 60], [120, 80], [115, 80]], "text": "上", "confidence": 0.9},
            {"polygon": [[200, 200], [205, 200], [205, 210], [200, 210]], "text": "外", "confidence": 0.9},
        ]

        count = _assign_paddle_predictions(column, predictions)

        self.assertEqual(count, 2)
        self.assertEqual(column["paddle_text"], "上下")
        self.assertAlmostEqual(column["paddle_confidence"], 0.925, places=3)

    def test_acceptance_rules(self):
        self.assertTrue(_text_acceptable("探知できている", 0.6))
        self.assertFalse(_text_acceptable("探知できている", 0.4))
        self.assertTrue(_text_acceptable("あ", 0.95))
        self.assertFalse(_text_acceptable("あ", 0.85))
        self.assertFalse(_text_acceptable("", 0.99))
        self.assertFalse(_text_acceptable("ABC123", 0.99))

    def test_finalize_keeps_paddle_result_when_acceptable(self):
        column = self._column([100, 50, 120, 150])
        column["paddle_text"] = "複製体"
        column["paddle_confidence"] = 0.8

        self.assertTrue(_finalize_paddle_column(column, "paddle-masked"))
        self.assertEqual(column["text"], "複製体")
        self.assertEqual(column["text_source"], "paddle-masked")

    def test_finalize_clears_unacceptable_result(self):
        column = self._column([100, 50, 120, 150])
        column["paddle_text"] = "あ"
        column["paddle_confidence"] = 0.5

        self.assertFalse(_finalize_paddle_column(column, "paddle-masked"))
        self.assertEqual(column["text"], "")


class ViewportRoundTripTests(unittest.TestCase):
    def test_image_to_display_inverts_display_to_image(self):
        image_size = (985, 1400)
        viewport = (663, 942)
        display = _image_bbox_to_display([77, 58, 116, 247], viewport, image_size)
        mapped = _display_bbox_to_image(display, viewport, image_size)

        self.assertAlmostEqual(mapped[0], 77, delta=2)
        self.assertAlmostEqual(mapped[1], 58, delta=2)
        self.assertAlmostEqual(mapped[2], 116, delta=2)
        self.assertAlmostEqual(mapped[3], 247, delta=2)


if __name__ == "__main__":
    unittest.main()
