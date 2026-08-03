import unittest
import json
from pathlib import Path

from photos_ocr import OcrColumn, group_columns, map_display_bbox, normalize_manifest


class PhotosOcrTests(unittest.TestCase):
    def test_maps_display_selection_back_to_source_pixels(self):
        result = map_display_bbox(
            display_bbox=(150, 100, 250, 300),
            image_viewport=(100, 50, 600, 850),
            image_size=(1000, 1600),
        )
        self.assertEqual(result, (100.0, 100.0, 300.0, 500.0))

    def test_groups_vertical_columns_right_to_left_before_translation(self):
        columns = [
            OcrColumn("c3", "受験者よりも", (410, 100, 438, 300), selection_order=2),
            OcrColumn("c1", "探知できている", (480, 100, 508, 300), selection_order=0),
            OcrColumn("c4", "僅かに少ないです。", (375, 100, 403, 300), selection_order=3),
            OcrColumn("c2", "複製体の総数が", (445, 100, 473, 300), selection_order=1),
        ]

        regions = group_columns(columns)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["column_ids"], ["c1", "c2", "c3", "c4"])
        self.assertEqual(
            regions[0]["text"],
            "探知できている複製体の総数が受験者よりも僅かに少ないです。",
        )
        self.assertTrue(regions[0]["needs_review"])
        self.assertEqual(len(regions[0]["source_segments"]), 4)

    def test_container_id_prevents_adjacent_bubbles_from_merging(self):
        columns = [
            OcrColumn("a1", "煙幕の", (500, 80, 530, 230), container_id="bubble-a"),
            OcrColumn("a2", "隙に", (465, 80, 495, 230), container_id="bubble-a"),
            OcrColumn("b1", "槍に塗布", (430, 80, 460, 230), container_id="bubble-b"),
            OcrColumn("b2", "していた", (395, 80, 425, 230), container_id="bubble-b"),
        ]

        regions = group_columns(columns)

        self.assertEqual([region["text"] for region in regions], ["煙幕の隙に", "槍に塗布していた"])
        self.assertTrue(all(region["grouping_basis"] == "container" for region in regions))
        self.assertTrue(all(not region["needs_review"] for region in regions))

    def test_splits_sentences_inside_one_bubble_at_column_terminal_punctuation(self):
        columns = [
            OcrColumn("c1", "探知できている", (280, 100, 310, 250), container_id="bubble-1"),
            OcrColumn("c2", "複製体の総数が", (245, 100, 275, 250), container_id="bubble-1"),
            OcrColumn("c3", "受験者よりも", (210, 100, 240, 250), container_id="bubble-1"),
            OcrColumn("c4", "僅かに少ないです。", (175, 100, 205, 280), container_id="bubble-1"),
            OcrColumn("c5", "特に一番の脅威である", (140, 130, 170, 320), container_id="bubble-1"),
            OcrColumn("c6", "ゼンゼさんの複製体の", (105, 130, 135, 320), container_id="bubble-1"),
            OcrColumn("c7", "反応が見つかりません。", (70, 130, 100, 320), container_id="bubble-1"),
        ]

        regions = group_columns(columns)

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            regions[0]["text"],
            "探知できている複製体の総数が受験者よりも僅かに少ないです。",
        )
        self.assertEqual(
            regions[1]["text"],
            "特に一番の脅威であるゼンゼさんの複製体の反応が見つかりません。",
        )
        self.assertEqual([region["sentence_index"] for region in regions], [1, 2])

    def test_manual_break_handles_missing_punctuation(self):
        columns = [
            OcrColumn("c1", "第一文", (80, 10, 100, 90), container_id="b", break_after=True),
            OcrColumn("c2", "第二文", (55, 10, 75, 90), container_id="b"),
        ]

        regions = group_columns(columns)

        self.assertEqual([region["text"] for region in regions], ["第一文", "第二文"])

    def test_skips_explicit_ruby_and_sfx_but_keeps_them_in_manifest(self):
        manifest = {
            "source": {"width": 1000, "height": 1600},
            "capture": {"image_viewport": [10, 20, 510, 820]},
            "columns": [
                {"id": "body", "text": "魔\n力\nを", "display_bbox": [110, 120, 130, 220]},
                {"id": "ruby", "text": "まりょく", "display_bbox": [132, 120, 142, 180], "role": "ruby"},
                {"id": "sfx", "text": "ぐっ", "display_bbox": [300, 300, 340, 420], "role": "sfx"},
            ],
        }

        result = normalize_manifest(manifest)

        self.assertEqual([column["text"] for column in result["columns"]], ["魔力を", "まりょく", "ぐっ"])
        self.assertEqual([region["text"] for region in result["regions"]], ["魔力を"])
        self.assertEqual(result["columns"][0]["image_bbox"], [200.0, 200.0, 240.0, 400.0])

    def test_page5_calibration_maps_seven_columns_to_two_sentences(self):
        sample_path = Path(__file__).parents[1] / "samples" / "photos_ocr_page5_excerpt.json"
        with sample_path.open("r", encoding="utf-8") as handle:
            result = normalize_manifest(json.load(handle))

        self.assertEqual(len(result["columns"]), 7)
        self.assertEqual(len(result["regions"]), 2)
        self.assertEqual(
            [region["text"] for region in result["regions"]],
            [
                "探知できている複製体の総数が受験者よりも僅かに少ないです。",
                "特に一番の脅威であるゼンゼさんの複製体の反応が見つかりません。",
            ],
        )
        first_box = result["columns"][0]["image_bbox"]
        self.assertTrue(290 < first_box[0] < 305)
        self.assertTrue(700 < first_box[3] < 710)


if __name__ == "__main__":
    unittest.main()
