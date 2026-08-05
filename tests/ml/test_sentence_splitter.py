from __future__ import annotations

import unittest

from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_dry_run import OcrLine
from manga_localizer.sentence_splitter import (
    assign_lines_to_islands,
    cluster_lines_into_columns,
    _detect_direction,
    split_island_dataset,
    split_sentences,
)


def _line(x1, y1, x2, y2, text, confidence=None):
    return OcrLine(
        polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        text=text,
        confidence=confidence,
    )


def _dataset(islands, size=(400, 400)):
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": list(size),
        "contours": True,
        "islands": islands,
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


class AssignLinesTest(unittest.TestCase):
    def test_assigns_and_flags_unassigned(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        assigned, unassigned, conflicts, violations = assign_lines_to_islands(
            dataset,
            [_line(20, 20, 40, 90, "ああ"), _line(200, 200, 220, 250, "外れ")],
        )
        self.assertEqual([a.island_id for a in assigned], ["island_001"])
        self.assertEqual([line.text for line in unassigned], ["外れ"])
        self.assertEqual(conflicts, ())
        self.assertEqual(violations, ())

    def test_polygon_outside_line_assigned_with_soft_violation(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                    "image_polygon": [[20, 10], [90, 10], [90, 90], [20, 90]],
                }
            ]
        )
        # center inside bbox but outside the ragged polygon -> still assigned,
        # recorded as a polygon violation (polygon is display-only geometry)
        assigned, unassigned, _, violations = assign_lines_to_islands(
            dataset,
            [_line(15, 40, 18, 60, "角"), _line(30, 30, 50, 80, "内")],
        )
        self.assertEqual([a.island_id for a in assigned], ["island_001", "island_001"])
        self.assertEqual(len(unassigned), 0)
        self.assertEqual([line.text for line, _ in violations], ["角"])
        self.assertEqual([island for _, island in violations], ["island_001"])

    def test_overlapping_islands_report_conflict(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 60, 60],
                    "image_center": [35, 35],
                    "image_area": 1.0,
                },
                {
                    "id": "island_002",
                    "image_bbox": [50, 50, 100, 100],
                    "image_center": [75, 75],
                    "image_area": 1.0,
                },
            ]
        )
        _, _, conflicts, _ = assign_lines_to_islands(dataset, [_line(52, 52, 58, 58, "重複")])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0][1], ("island_001", "island_002"))


class DirectionTest(unittest.TestCase):
    def test_horizontal_detected(self):
        lines = [
            _line(10, 20, 60, 30, "a"),
            _line(70, 20, 120, 30, "b"),
            _line(130, 20, 180, 30, "c"),
        ]
        self.assertTrue(_detect_direction(lines))

    def test_vertical_stack_detected(self):
        lines = [
            _line(20, 10, 30, 60, "a"),
            _line(20, 70, 30, 120, "b"),
            _line(20, 130, 30, 180, "c"),
        ]
        self.assertFalse(_detect_direction(lines))

    def test_single_line_by_aspect(self):
        self.assertTrue(_detect_direction([_line(10, 10, 90, 20, "a")]))
        self.assertFalse(_detect_direction([_line(40, 10, 50, 90, "a")]))


class ClusteringTest(unittest.TestCase):
    def test_vertical_columns_merge_by_x_band(self):
        lines = [
            _line(20, 10, 30, 60, "一"),
            _line(20, 70, 30, 120, "二"),
            _line(60, 10, 70, 60, "三"),
        ]
        columns = cluster_lines_into_columns(lines, vertical=True)
        self.assertEqual(len(columns), 2)
        self.assertEqual(columns[0].text, "一二")
        self.assertEqual(columns[1].text, "三")

    def test_horizontal_rows_merge_by_y_band(self):
        lines = [
            _line(10, 20, 60, 30, "a"),
            _line(70, 20, 120, 30, "b"),
            _line(10, 50, 60, 60, "c"),
        ]
        columns = cluster_lines_into_columns(lines, vertical=False)
        self.assertEqual(len(columns), 2)
        self.assertEqual(columns[0].text, "ab")


class SplitSentencesTest(unittest.TestCase):
    def test_strong_end_char_splits(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        columns = [
            type("C", (), {"text": "それは…", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
            type("C", (), {"text": "違うんだ。", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
        ]
        sentences = split_sentences(island, columns, full_text="それは…違うんだ。")
        self.assertEqual(len(sentences), 2)
        self.assertTrue(all(s.boundary_signal == "strong" for s in sentences))
        self.assertEqual(sentences[0].text, "それは…")

    def test_mid_line_strong_char_splits(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        columns = [
            type("C", (), {"text": "行こう。次は", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
        ]
        sentences = split_sentences(island, columns, full_text="行こう。次は")
        self.assertEqual(len(sentences), 2)
        self.assertEqual([s.text for s in sentences], ["行こう。", "次は"])

    def test_no_signal_flags_review(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        columns = [
            type("C", (), {"text": "ワテクア", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
        ]
        sentences = split_sentences(island, columns, full_text="ワテクア")
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].boundary_signal, "low")
        self.assertTrue(sentences[0].needs_review)

    def test_particle_end_sets_signal_without_split(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        columns = [
            type("C", (), {"text": "魔力の性質を", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
            type("C", (), {"text": "弄らせたのですね", "bbox": (0, 0, 1, 1), "direction": "vertical"})(),
        ]
        sentences = split_sentences(island, columns, full_text="魔力の性質を弄らせたのですね")
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0].boundary_signal, "particle")
        self.assertFalse(sentences[0].needs_review)

    def _column(self, text, lines):
        return type("C", (), {"text": text, "bbox": (0, 0, 1, 1), "direction": "vertical", "lines": lines})()

    def test_low_confidence_punctuation_does_not_split(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        column = self._column(
            "行こう。次は",
            [_line(10, 10, 20, 40, "行こう。次は", confidence=0.5)],
        )
        sentences = split_sentences(island, [column], full_text="行こう。次は")
        self.assertEqual(len(sentences), 1)
        self.assertTrue(sentences[0].needs_review)
        self.assertEqual(sentences[0].boundary_signal, "low")

    def test_high_confidence_punctuation_splits(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        column = self._column(
            "行こう。次は",
            [_line(10, 10, 20, 40, "行こう。次は", confidence=0.95)],
        )
        sentences = split_sentences(island, [column], full_text="行こう。次は")
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].boundary_signal, "strong")
        self.assertFalse(sentences[0].needs_review)
        # trailing fragment without an end signal stays flagged for review
        self.assertTrue(sentences[1].needs_review)

    def test_no_confidence_punctuation_splits_directly(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                }
            ]
        )
        island = dataset.islands[0]
        column = self._column(
            "行こう。次は",
            [_line(10, 10, 20, 40, "行こう。次は")],
        )
        sentences = split_sentences(island, [column], full_text="行こう。次は")
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].boundary_signal, "strong")
        self.assertFalse(sentences[0].needs_review)


class EndToEndSplitTest(unittest.TestCase):
    def test_full_split_result(self):
        dataset = _dataset(
            [
                {
                    "id": "island_001",
                    "image_bbox": [10, 10, 100, 100],
                    "image_center": [55, 55],
                    "image_area": 1.0,
                },
                {
                    "id": "island_002",
                    "image_bbox": [200, 10, 300, 100],
                    "image_center": [250, 55],
                    "image_area": 1.0,
                },
            ]
        )
        lines = [
            _line(20, 10, 30, 130, "左列"),
            _line(50, 10, 60, 130, "右列"),
            _line(210, 10, 220, 130, "二島"),
            _line(150, 10, 160, 130, "外"),
        ]
        result = split_island_dataset(dataset, lines)
        self.assertEqual(len(result.units), 2)
        self.assertEqual([u.island.id for u in result.units], ["island_002", "island_001"])
        self.assertEqual(len(result.unassigned), 1)
        # vertical reading order: right column first
        self.assertEqual(result.units[1].full_text, "右列左列")
        self.assertEqual(result.units[0].full_text, "二島")


if __name__ == "__main__":
    unittest.main()
