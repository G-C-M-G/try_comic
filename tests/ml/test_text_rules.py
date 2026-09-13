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
