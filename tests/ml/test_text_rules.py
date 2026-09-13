"""跨路径共用的文本规则（省略号归一）。"""

from __future__ import annotations

import unittest

from manga_localizer.text_rules import normalize_ellipsis


class NormalizeEllipsisTest(unittest.TestCase):
    def test_six_dots_collapse_to_three(self):
        self.assertEqual(normalize_ellipsis("……"), "…")
        self.assertEqual(normalize_ellipsis("はい……"), "はい…")
        self.assertEqual(normalize_ellipsis("ああ……！！"), "ああ…！！")

    def test_fullwidth_and_ideographic_dot_runs(self):
        self.assertEqual(normalize_ellipsis("．．．"), "…")
        self.assertEqual(normalize_ellipsis("．．．．．"), "…")
        self.assertEqual(normalize_ellipsis("。。。"), "…")

    def test_ascii_dots_three_or_more(self):
        self.assertEqual(normalize_ellipsis("..."), "…")
        self.assertEqual(normalize_ellipsis("あ..."), "あ…")

    def test_two_dots_are_left_alone(self):
        """2 点不改成 3 点（避免把本来两点写成三点）。"""
        self.assertEqual(normalize_ellipsis(".."), "..")
        self.assertEqual(normalize_ellipsis("．．"), "．．")

    def test_two_ellipsis_chars_collapse(self):
        """两个 U+2026 就是日漫的 6 点写法 → 3 点。"""
        self.assertEqual(normalize_ellipsis("……"), "…")

    def test_plain_text_untouched(self):
        self.assertEqual(normalize_ellipsis("普通文本"), "普通文本")
        self.assertEqual(normalize_ellipsis(""), "")


if __name__ == "__main__":
    unittest.main()


class ShortSourceTests(unittest.TestCase):
    """短原文判据：≤ N 字（非空白计数），0/负数 = 关闭。"""

    def test_short_kana_and_kanji_lines(self):
        from manga_localizer.text_rules import short_source

        self.assertTrue(short_source("ぎゅっ♡", 6))
        self.assertTrue(short_source("甲书", 6))
        self.assertTrue(short_source("あ", 6))

    def test_long_source_not_short(self):
        from manga_localizer.text_rules import short_source

        self.assertFalse(short_source("これはテスト用の長文", 6))
        self.assertFalse(short_source("あ" * 7, 6))

    def test_whitespace_not_counted(self):
        from manga_localizer.text_rules import short_source

        self.assertTrue(short_source("ぎゅ っ ♡", 4))
        self.assertFalse(short_source("ぎゅ っ ♡", 3))

    def test_threshold_is_adjustable(self):
        from manga_localizer.text_rules import short_source

        self.assertFalse(short_source("ぎゅっ", 2))
        self.assertTrue(short_source("ぎゅっ", 3))

    def test_zero_disables(self):
        from manga_localizer.text_rules import short_source

        self.assertFalse(short_source("あ", 0))
        self.assertFalse(short_source("あ", -1))

    def test_empty_is_not_short(self):
        from manga_localizer.text_rules import short_source

        self.assertFalse(short_source("", 6))
        self.assertFalse(short_source("   ", 6))


class OcrCommentaryTests(unittest.TestCase):
    """AI OCR 的说明性输出不是读数（实测：被当读数翻成中文塞进原框）。"""

    def test_real_observed_samples(self):
        from manga_localizer.text_rules import is_ocr_commentary

        self.assertTrue(is_ocr_commentary("图片中没有任何日文文本。它看起来像是…"))
        self.assertTrue(is_ocr_commentary("图片中没有文字。"))

    def test_other_description_forms(self):
        from manga_localizer.text_rules import is_ocr_commentary

        self.assertTrue(is_ocr_commentary("无法辨认"))
        self.assertTrue(is_ocr_commentary("以下是图片中的文字："))
        self.assertTrue(is_ocr_commentary("no text in this image"))
        self.assertTrue(is_ocr_commentary("There is no Japanese text here."))

    def test_normal_readings_pass(self):
        from manga_localizer.text_rules import is_ocr_commentary

        self.assertFalse(is_ocr_commentary("ぎゅっ♡"))
        self.assertFalse(is_ocr_commentary("これはテスト用の長文"))
        self.assertFalse(is_ocr_commentary("图片中有一只猫"))   # 不是"没有文字"
        self.assertFalse(is_ocr_commentary(""))
