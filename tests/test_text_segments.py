import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "manga-image-translator"))

from manga_translator.rendering import expand_translation_regions
from manga_translator.utils import TextBlock
from text_segments import (
    attach_translation_segments,
    group_regions_into_sentences,
    has_sentence_terminal,
    split_translation_for_segments,
)
from text_paint import normalize_vertical_text


def block(text, bbox, panel, font_size=24):
    x1, y1, x2, y2 = bbox
    region = TextBlock(
        lines=np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]),
        texts=[text],
        font_size=font_size,
        fg_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        direction="v",
    )
    region.panel_index = panel
    return region


class SentenceGroupingTests(unittest.TestCase):
    def test_vertical_ellipsis_uses_one_presentation_glyph(self):
        self.assertEqual(normalize_vertical_text("不行了……"), "不行了︙")
        self.assertEqual(normalize_vertical_text("ダメだ．．．"), "ダメだ︙")

    def test_terminal_punctuation_includes_japanese_period_and_ellipsis(self):
        self.assertTrue(has_sentence_terminal("終わり。"))
        self.assertTrue(has_sentence_terminal("ダメだ．．．"))
        self.assertTrue(has_sentence_terminal("本当！？"))
        self.assertFalse(has_sentence_terminal("まだ続く"))

    def test_page1_groups_by_panel_until_sentence_terminal(self):
        regions = [
            block("煙幕の隙に", (544, 70, 623, 169), 0),
            block("槍に塗布していたのか！？", (293, 71, 406, 199), 0),
            block("やられた！！", (46, 58, 95, 263), 1),
            block("ダメだ．．．", (501, 417, 534, 569), 2),
            block("体だけじゃなく魔力を通す経路も", (222, 453, 397, 622), 2),
            block("完全に麻痺している", (51, 487, 170, 736), 2),
            block("それなのに思考だけが", (755, 1081, 829, 1247), 3),
            block("こんなにもハッキリして．．．", (64, 1086, 138, 1318), 3),
        ]

        grouped = group_regions_into_sentences(regions)

        self.assertEqual(
            [region.text for region in grouped],
            [
                "煙幕の隙に槍に塗布していたのか！？",
                "やられた！！",
                "ダメだ．．．",
                "体だけじゃなく魔力を通す経路も完全に麻痺している",
                "それなのに思考だけがこんなにもハッキリして．．．",
            ],
        )
        self.assertEqual(len(grouped[0].render_segments), 2)
        self.assertEqual(len(grouped[3].render_segments), 2)
        self.assertEqual(
            grouped[0].translation_query,
            "<1>煙幕の隙に<2>槍に塗布していたのか！？",
        )

    def test_translation_is_split_back_to_render_boxes(self):
        region = group_regions_into_sentences([
            block("体だけじゃなく魔力を通す経路も", (222, 453, 397, 622), 2, 22),
            block("完全に麻痺している", (51, 487, 170, 736), 2, 39),
        ])[0]
        region.translation = "不只是身体，连传导魔力的路径也完全麻痹了。"

        segments = attach_translation_segments(region)
        expanded = expand_translation_regions([region])

        self.assertEqual(segments, ["不只是身体，", "连传导魔力的路径也完全麻痹了。"])
        self.assertEqual([item.translation for item in expanded], segments)
        self.assertEqual([item.font_size for item in expanded], [22, 39])
        self.assertEqual([len(item.lines) for item in expanded], [1, 1])

    def test_proportional_fallback_returns_one_part_per_box(self):
        parts = split_translation_for_segments("身体和魔力通路都彻底麻痹了。", ["体だけじゃなく", "魔力を通す経路も", "完全に麻痺している"])
        self.assertEqual(len(parts), 3)
        self.assertEqual("".join(parts), "身体和魔力通路都彻底麻痹了。")

    def test_translation_markers_take_priority_over_length_split(self):
        parts = split_translation_for_segments(
            "<1>然而只有思绪却<2>如此清晰……",
            ["それなのに思考だけが", "こんなにもハッキリして．．．"],
        )
        self.assertEqual(parts, ["然而只有思绪却", "如此清晰……"])

    def test_missing_panel_ids_do_not_merge_distant_regions(self):
        first = block("第一段还没句号", (800, 100, 880, 260), None)
        second = block("另一个气泡。", (80, 700, 180, 900), None)

        grouped = group_regions_into_sentences([first, second])

        self.assertEqual([item.text for item in grouped], ["第一段还没句号", "另一个气泡。"])

    def test_repeated_ellipsis_never_becomes_a_punctuation_only_box(self):
        parts = split_translation_for_segments(
            "可是思绪却如此清晰……",
            ["それなのに思考だけが", "こんなにもハッキリして．．．"],
        )
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(part.strip("，、；：。！？!?…．.") for part in parts))


if __name__ == "__main__":
    unittest.main()
