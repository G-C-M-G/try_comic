from __future__ import annotations

import unittest

from manga_localizer.cross_validate import (
    _normalized_edit_distance,
    cross_validate_page,
)
from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_dry_run import OcrLine


def _line(x1, y1, x2, y2, text, confidence=None, island_id=None):
    line = OcrLine(
        polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        text=text,
        confidence=confidence,
    )
    if island_id:
        object.__setattr__(line, "island_id", island_id)
    return line


def _dataset(islands, size=(400, 400)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


class EditDistanceTest(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(_normalized_edit_distance("あいうえお", "あいうえお"), 0.0)

    def test_empty_pairs(self):
        self.assertEqual(_normalized_edit_distance("", ""), 0.0)
        self.assertEqual(_normalized_edit_distance("あ", ""), 1.0)

    def test_diverge_is_high(self):
        distance = _normalized_edit_distance("ヨドインが", "リョドインが！！")
        self.assertGreater(distance, 0.3)


class CrossValidateTest(unittest.TestCase):
    def _dataset(self):
        return _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 110, 110],
                    "image_center": [60, 60],
                    "image_area": 5000.0,
                },
                {
                    "id": "island_002",
                    "image_bbox": [200, 10, 300, 100],
                    "image_center": [250, 55],
                    "image_area": 1000.0,
                },
                {
                    "id": "island_003",
                    "image_bbox": [20, 200, 60, 240],
                    "image_center": [40, 220],
                    "image_area": 100.0,
                },
            ]
        )

    def test_pass_when_agree(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "一致する", island_id="island_001")]
        paddle = [
            _line(15, 15, 40, 90, "一致", 0.99, island_id="island_001"),
            _line(45, 15, 70, 90, "する", 0.98, island_id="island_001"),
        ]
        result = cross_validate_page(dataset, manga, paddle)
        by_id = {v.island_id: v for v in result.verdicts}
        self.assertEqual(by_id["island_001"].status, "pass")
        self.assertEqual(by_id["island_003"].status, "dropped_tiny")

    def test_review_on_divergence(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "ヨドインが", island_id="island_001")]
        paddle = [_line(15, 15, 40, 90, "リョドインが", 0.99, island_id="island_001")]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "review")
        self.assertIn("engines diverge", " ".join(verdict.review_reasons))

    def test_low_confidence_passes_when_texts_agree(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "そうで", island_id="island_001")]
        paddle = [_line(15, 15, 40, 90, "そうで", 0.31, island_id="island_001")]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "pass")
        self.assertEqual(len(verdict.low_confidence_lines), 1)

    def test_low_confidence_reviews_on_divergence(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "そうで", island_id="island_001")]
        paddle = [_line(15, 15, 40, 90, "そうだ", 0.31, island_id="island_001")]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "review")
        self.assertIn("low-confidence", " ".join(verdict.review_reasons))

    def test_review_when_manga_empty(self):
        dataset = self._dataset()
        paddle = [_line(15, 15, 40, 90, "テキスト", 0.95, island_id="island_002")]
        result = cross_validate_page(dataset, [], paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_002"]
        self.assertEqual(verdict.status, "review")
        self.assertIn("manga-ocr empty", " ".join(verdict.review_reasons))
        self.assertFalse(verdict.empty_paddle)

    def test_pass_when_fragments_out_of_order(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "自分の軍だけでも容易く人間界を制圧することが可能だ」", island_id="island_001")]
        paddle = [
            _line(15, 15, 40, 90, "可能だ』", 0.99, island_id="island_001"),
            _line(45, 15, 70, 90, "制圧することが", 0.98, island_id="island_001"),
            _line(75, 15, 100, 90, "容易く人間界を", 0.97, island_id="island_001"),
            _line(20, 40, 60, 100, "自分の軍だけでも", 0.96, island_id="island_001"),
        ]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "pass")
        self.assertEqual(verdict.edit_distance, 0.0)

    def test_review_when_fragment_not_contained(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "ヨドインが", island_id="island_001")]
        paddle = [_line(15, 15, 40, 90, "リョドインが", 0.99, island_id="island_001")]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "review")

    def test_review_when_only_short_fragments(self):
        dataset = self._dataset()
        manga = [_line(10, 10, 110, 110, "ワテクアは", island_id="island_001")]
        paddle = [_line(15, 15, 40, 90, "は", 0.99, island_id="island_001")]
        result = cross_validate_page(dataset, manga, paddle)
        verdict = {v.island_id: v for v in result.verdicts}["island_001"]
        self.assertEqual(verdict.status, "review")
        self.assertGreater(verdict.edit_distance, 0.3)

    def test_overlapping_bbox_assigns_to_smallest_island(self):
        # a tiny island's line falls inside a neighbour's expanded bbox:
        # the manga text must land on the tiny island, not overwrite the big one
        dataset = _dataset(
            [
                {
                    "id": "island_big",
                    "image_bbox": [10, 10, 110, 110],
                    "image_center": [60, 60],
                    "image_area": 9000.0,
                },
                {
                    "id": "island_tiny",
                    "image_bbox": [20, 20, 40, 40],
                    "image_center": [30, 30],
                    "image_area": 900.0,
                },
            ]
        )
        manga = [
            _line(10, 10, 110, 110, "重要な文", island_id="island_big"),
            _line(20, 20, 40, 40, "．．．", island_id="island_tiny"),
        ]
        result = cross_validate_page(dataset, manga, [])
        by_id = {v.island_id: v for v in result.verdicts}
        self.assertEqual(by_id["island_big"].manga_text, "重要な文")
        self.assertEqual(by_id["island_tiny"].manga_text, "．．．")


if __name__ == "__main__":
    unittest.main()
