import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "manga-image-translator"))

from manga_translator.ocr.model_manga_ocr import prefer_axis_aligned_dakuten


class MangaOcrDakutenTests(unittest.TestCase):
    def test_restores_missing_dakuten(self):
        self.assertEqual(
            prefer_axis_aligned_dakuten("タメだ．．．", "ダメだ．．．"),
            "ダメだ．．．",
        )

    def test_does_not_replace_unrelated_characters(self):
        self.assertEqual(prefer_axis_aligned_dakuten("ゼノゼ", "ゼンゼ"), "ゼノゼ")
        self.assertEqual(prefer_axis_aligned_dakuten("煙幕", "煙霧"), "煙幕")

    def test_does_not_remove_existing_dakuten(self):
        self.assertEqual(prefer_axis_aligned_dakuten("ダメ", "タメ"), "ダメ")


if __name__ == "__main__":
    unittest.main()
