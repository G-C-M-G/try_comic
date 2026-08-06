import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from manga_localizer.cross_validate import (
    _normalize_for_match,
    _split_foreign_fragments,
    cross_validate_page,
)
from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_dry_run import OcrLine


def _line(x1, y1, x2, y2, text, confidence=None):
    return OcrLine(
        polygon=((float(x1), float(y1)), (float(x2), float(y1)),
                 (float(x2), float(y2)), (float(x1), float(y2))),
        text=text,
        confidence=confidence,
    )


def _dataset(islands, tmp):
    payload = {
        "schema_version": 1,
        "page": "99",
        "image_size": [800, 1200],
        "islands": [
            {
                "id": island_id,
                "image_bbox": list(bbox),
                "image_center": [
                    (bbox[0] + bbox[2]) // 2,
                    (bbox[1] + bbox[3]) // 2,
                ],
                "image_area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            }
            for island_id, bbox in islands
        ],
    }
    return dataset_from_data(payload, path=Path(tmp) / "islands.json",
                             source_sha256="test")


class MatchNormalizationTests(unittest.TestCase):
    def test_normalizes_small_kana(self):
        self.assertEqual(_normalize_for_match("まぁ数回"), "まあ数回")
        self.assertEqual(_normalize_for_match("ちょっと"), "ちよつと")

    def test_strips_punctuation(self):
        self.assertEqual(_normalize_for_match("ああ．．．？"), "ああ")
        self.assertEqual(_normalize_for_match("．．．？"), "")


class PunctuationIslandTests(unittest.TestCase):
    def test_punctuation_only_island_passes_without_paddle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(
                [("island_001", (400, 100, 480, 200))], tmp
            )
            manga = [_line(400, 100, 480, 200, "．．．？")]
            result = cross_validate_page(dataset, manga, [])
            self.assertEqual(result.verdicts[0].status, "pass")
            self.assertEqual(result.review_count, 0)


class SmallKanaCrossCheckTests(unittest.TestCase):
    def test_small_kana_divergence_is_now_agreement(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(
                [("island_001", (400, 100, 480, 200))], tmp
            )
            manga = [_line(400, 100, 480, 200, "まぁ数回打ち合った")]
            paddle = [
                _line(402, 110, 430, 180, "まあ", 0.99),
                _line(432, 110, 470, 180, "数回", 0.99),
            ]
            result = cross_validate_page(dataset, manga, paddle)
            self.assertEqual(result.verdicts[0].status, "pass")


class ForeignFragmentTests(unittest.TestCase):
    def test_foreign_column_excluded_from_divergence(self):
        # manga reads only the name column; paddle also reads the neighbour
        # column that does not x-overlap the evidence
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(
                [("island_001", (100, 100, 400, 500))], tmp
            )
            manga = [_line(100, 100, 400, 500, "ウイラスは")]
            paddle = [
                _line(120, 120, 250, 450, "ウイラスは", 0.99),
                _line(300, 130, 380, 400, "純然たる光", 0.99),
            ]
            result = cross_validate_page(dataset, manga, paddle)
            verdict = result.verdicts[0]
            self.assertEqual(verdict.status, "pass")
            self.assertEqual(verdict.excluded_foreign, ("純然たる光",))

    def test_overlapping_misread_still_diverges(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(
                [("island_001", (100, 100, 400, 500))], tmp
            )
            manga = [_line(100, 100, 400, 500, "ワテクアは")]
            paddle = [_line(110, 120, 380, 450, "デクは", 0.95)]
            result = cross_validate_page(dataset, manga, paddle)
            self.assertEqual(result.verdicts[0].status, "review")


class RegressionTests(unittest.TestCase):
    def test_verdict_schema_fields_intact(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(
                [("island_001", (400, 100, 480, 200))], tmp
            )
            result = cross_validate_page(dataset, [], [])
            payload = {
                "island_id": "island_001",
                "status": "review",
                "manga_text": "",
                "paddle_columns": [],
                "edit_distance": None,
                "low_confidence_lines": [],
                "review_reasons": ["manga-ocr empty", "paddle empty"],
                "dropped_reason": None,
                "excluded_foreign": [],
            }
            self.assertEqual(payload["island_id"], result.verdicts[0].island_id)
            self.assertEqual(payload["review_reasons"], list(result.verdicts[0].review_reasons))


if __name__ == "__main__":
    unittest.main()
