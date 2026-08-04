import json
from pathlib import Path
import tempfile
import unittest

from background_ocr import attach_manual_review_queue, manual_review_reasons
from ocr_review import apply_review, build_review_template


def _column(column_id, text, agrees, source="cross-model-agreement"):
    return {
        "id": column_id,
        "text": text,
        "polygon": [[10, 10], [30, 10], [30, 100], [10, 100]],
        "image_bbox": [10, 10, 30, 100],
        "layout_bbox_10pct": [9, 5, 31, 105],
        "selection_order": 0,
        "panel_index": 0,
        "detection_sources": ["ctd", "paddleocr"],
        "paddle_validation": [{
            "text": text if agrees else "別の文字",
            "confidence": 0.98,
            "box_overlap": 0.95,
        }],
        "readings": [text, text, text],
        "text_source": source,
        "recognition_confidence": 1.0,
        "cross_validation": {"agrees": agrees},
    }


class ManualReviewTests(unittest.TestCase):
    def test_cross_model_disagreement_requires_review(self):
        column = _column("column_001", "本文", False, "manga_ocr")

        reasons = manual_review_reasons(column)

        self.assertIn("cross-model-text-disagreement", reasons)

    def test_exact_cross_model_agreement_does_not_require_review(self):
        result = {"columns": [_column("column_001", "本文", True)], "fusion_summary": {}}

        attach_manual_review_queue(result)

        self.assertFalse(result["manual_review"]["translation_blocked"])
        self.assertFalse(result["columns"][0]["review_required"])

    def test_single_model_detection_requires_review(self):
        column = _column("column_001", "本文", None, "manga_ocr")
        column["detection_sources"] = ["ctd"]
        column["paddle_validation"] = []

        reasons = manual_review_reasons(column)

        self.assertIn("single-model-detection", reasons)
        self.assertIn("missing-cross-model-text-confirmation", reasons)

    def test_manual_review_can_be_bypassed_without_losing_candidates(self):
        result = {
            "columns": [_column("column_001", "本文", False, "manga_ocr")],
            "fusion_summary": {},
        }

        attach_manual_review_queue(result, enabled=False)

        self.assertFalse(result["manual_review"]["translation_blocked"])
        self.assertEqual(result["manual_review"]["status"], "bypassed")
        self.assertEqual(result["manual_review"]["candidate_count"], 1)
        self.assertEqual(result["manual_review_queue"][0]["status"], "bypassed")

    def test_paddle_geometry_can_expand_an_incomplete_current_box(self):
        column = _column("column_001", "本文", True)
        column["paddle_validation"][0].update({
            "polygon": [[8, 8], [32, 8], [32, 140], [8, 140]],
            "image_bbox": [8, 8, 32, 140],
        })
        result = {
            "schema_version": 4,
            "source": {"path": "1.jpg", "width": 200, "height": 200},
            "columns": [column],
            "reading_order": ["column_001"],
            "translation_units": [{
                "id": "sentence_001",
                "column_ids": ["column_001"],
                "source_text": "本文",
                "translation_query": "本文",
                "panel_index": 0,
                "layout_regions": [],
            }],
            "fusion_summary": {},
        }
        attach_manual_review_queue(result)

        self.assertIn(
            "paddle-box-extends-current-region",
            result["columns"][0]["review_reasons"],
        )
        self.assertEqual(
            result["manual_review_queue"][0]["paddle_image_bbox"],
            [8, 8, 32, 140],
        )

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.ocr.json"
            review = Path(temporary) / "source.review.json"
            output = Path(temporary) / "reviewed.ocr.json"
            source.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            template = build_review_template(result, source)
            template["items"][0]["decision"] = "corrected"
            template["items"][0]["use_paddle_geometry"] = True
            review.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")

            apply_review(source, review, output)
            reviewed = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(reviewed["columns"][0]["image_bbox"], [8, 8, 32, 140])
        self.assertEqual(reviewed["columns"][0]["layout_bbox_10pct"], [7, 1, 33, 147])

    def test_review_correction_rebuilds_sentence_text(self):
        result = {
            "schema_version": 4,
            "source": {"path": "1.jpg", "width": 200, "height": 200},
            "columns": [
                _column("column_001", "誤字", False, "manga_ocr"),
                _column("column_002", "です。", True),
            ],
            "reading_order": ["column_001", "column_002"],
            "translation_units": [{
                "id": "sentence_001",
                "column_ids": ["column_001", "column_002"],
                "source_text": "誤字です。",
                "translation_query": "<1>誤字<2>です。",
                "panel_index": 0,
                "layout_regions": [],
            }],
            "fusion_summary": {},
        }
        attach_manual_review_queue(result)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.ocr.json"
            review = Path(temporary) / "source.review.json"
            output = Path(temporary) / "reviewed.ocr.json"
            source.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            template = build_review_template(result, source)
            template["items"][0]["decision"] = "corrected"
            template["items"][0]["corrected_text"] = "本文"
            # Windows editors commonly save reviewed JSON with a UTF-8 BOM.
            review.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8-sig")

            apply_review(source, review, output)
            reviewed = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(reviewed["manual_review"]["translation_blocked"])
        self.assertEqual(reviewed["translation_units"][0]["source_text"], "本文です。")
        self.assertEqual(reviewed["columns"][0]["review_status"], "corrected")


if __name__ == "__main__":
    unittest.main()
