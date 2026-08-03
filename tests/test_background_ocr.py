import unittest

import numpy as np

from background_ocr import (
    benchmark_against_photos,
    canonical_benchmark_text,
    choose_cross_validated_text,
    choose_consensus,
    expand_bbox_10pct,
    fuse_paddle_detections,
    split_polygon_columns,
)


class BackgroundOcrTests(unittest.TestCase):
    def test_consensus_uses_medoid_and_similarity(self):
        text, confidence = choose_consensus(["槍に塗布", "槍に塗布", "槍に塗布し"])
        self.assertEqual(text, "槍に塗布")
        self.assertGreater(confidence, 0.9)

    def test_benchmark_punctuation_normalizes_equivalent_forms(self):
        self.assertEqual(canonical_benchmark_text("本当！？．．．"), "本当!?…")
        self.assertEqual(canonical_benchmark_text("本当!?…"), "本当!?…")

    def test_layout_bbox_allows_ten_percent_total_overflow(self):
        self.assertEqual(expand_bbox_10pct([10, 20, 110, 220], (500, 500)), [5, 10, 115, 230])

    def test_splits_wide_mask_into_vertical_columns(self):
        mask = np.zeros((150, 160), dtype=np.uint8)
        mask[20:120, 20:45] = 255
        mask[25:125, 70:95] = 255
        mask[30:130, 120:145] = 255
        polygon = np.array([[10, 10], [150, 10], [150, 140], [10, 140]])

        columns = split_polygon_columns(polygon, mask, mask.shape)

        self.assertEqual(len(columns), 3)

    def test_keeps_fragmented_narrow_vertical_region_as_one_column(self):
        mask = np.zeros((160, 80), dtype=np.uint8)
        mask[15:70, 20:37] = 255
        mask[80:145, 42:47] = 255
        polygon = np.array([[15, 10], [50, 10], [50, 150], [15, 150]])

        columns = split_polygon_columns(polygon, mask, mask.shape)

        self.assertEqual(len(columns), 1)
        np.testing.assert_array_equal(columns[0], polygon)

    def test_does_not_split_horizontal_text_into_characters(self):
        mask = np.zeros((80, 180), dtype=np.uint8)
        mask[25:55, 20:45] = 255
        mask[25:55, 70:95] = 255
        mask[25:55, 120:145] = 255
        polygon = np.array([[10, 15], [160, 15], [160, 65], [10, 65]])

        columns = split_polygon_columns(polygon, mask, mask.shape)

        self.assertEqual(len(columns), 1)
        np.testing.assert_array_equal(columns[0], polygon)

    def test_benchmark_requires_all_multi_character_columns(self):
        result = {
            "columns": [
                {"id": "column_001", "text": "槍に塗布", "image_bbox": [10, 10, 30, 100]}
            ]
        }
        photos = {
            "backend": "Windows Photos Scan Text",
            "columns": [
                {"id": "column_001", "text": "槍に塗布", "image_bbox": [9, 9, 31, 101]},
                {"id": "column_002", "text": "隙に", "image_bbox": [40, 10, 60, 80]},
            ],
        }

        report = benchmark_against_photos(result, photos)

        self.assertFalse(report["passes"]["no_missing_multi_character_columns"])
        self.assertEqual(report["missing_multi_character_columns"], ["column_002"])

    def test_box_match_rate_is_f1_not_recall_only(self):
        result = {
            "columns": [
                {"id": "one", "text": "本文", "image_bbox": [10, 10, 30, 100]},
                {"id": "extra", "text": "誤検出", "image_bbox": [80, 10, 110, 100]},
            ]
        }
        photos = {
            "backend": "Windows Photos Scan Text",
            "columns": [
                {"id": "reference", "text": "本文", "image_bbox": [10, 10, 30, 100]},
            ],
        }

        report = benchmark_against_photos(result, photos)

        self.assertEqual(report["text_box_precision"], 0.5)
        self.assertEqual(report["text_box_recall"], 1.0)
        self.assertEqual(report["text_box_match_rate"], 0.6667)

    def test_paddle_overlap_validates_without_replacing_ctd_geometry(self):
        ctd = [{
            "polygon": [[10, 10], [30, 10], [30, 100], [10, 100]],
            "image_bbox": [10, 10, 30, 100],
            "detection_confidence": 0.8,
            "direction": "vertical",
        }]
        paddle = [{
            "polygon": [[8, 8], [34, 8], [34, 104], [8, 104]],
            "text": "本文です。",
            "confidence": 0.99,
        }]

        fused, rejected = fuse_paddle_detections(ctd, paddle, (200, 200))

        self.assertFalse(rejected)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["image_bbox"], [10, 10, 30, 100])
        self.assertEqual(fused[0]["detection_sources"], ["ctd", "paddleocr"])

    def test_paddle_only_multichar_box_can_supplement_detection(self):
        paddle = [{
            "polygon": [[40, 10], [65, 10], [65, 100], [40, 100]],
            "text": "補足本文",
            "confidence": 0.96,
        }]

        fused, rejected = fuse_paddle_detections([], paddle, (200, 200))

        self.assertFalse(rejected)
        self.assertEqual(len(fused), 1)
        self.assertTrue(fused[0]["supplemental"])
        text, _, source, _ = choose_cross_validated_text(
            ["", "", ""], fused[0]["paddle_validation"]
        )
        self.assertEqual(text, "補足本文")
        self.assertEqual(source, "paddleocr-rescue")

    def test_paddle_only_single_character_is_not_rescued(self):
        text, confidence, source, _ = choose_cross_validated_text(
            ["", "", ""], [{"text": "し", "confidence": 0.99}]
        )

        self.assertEqual(text, "")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(source, "manga_ocr")

    def test_high_confidence_paddle_text_rescues_low_manga_consensus(self):
        text, confidence, source, comparison = choose_cross_validated_text(
            ["異なる本文", "別の文章", "違う文字"],
            [{"text": "正しい本文", "confidence": 0.97}],
        )

        self.assertEqual(text, "正しい本文")
        self.assertEqual(confidence, 0.97)
        self.assertEqual(source, "paddleocr-rescue")
        self.assertEqual(comparison["decision"], "paddleocr-low-manga-consensus")


if __name__ == "__main__":
    unittest.main()
