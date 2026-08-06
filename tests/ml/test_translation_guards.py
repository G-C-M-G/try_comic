import tempfile
import unittest
from pathlib import Path

from manga_localizer.errors import TranslationResponseError
from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_translate import (
    IslandTranslateConfig,
    IslandTranslationCache,
    MockIslandTranslator,
    _is_short_symbol,
    _validate_island_translation,
    translate_island_page,
)
from manga_localizer.translator import TranslationRequest, TranslationResponse


def _fresh_cache(tmp: str):
    return IslandTranslationCache(Path(tmp) / "cache")


def _dataset():
    data = {
        "schema_version": 1,
        "page": "synthetic",
        "image_size": [400, 400],
        "contours": True,
        "islands": [
            {
                "id": "island_001",
                "image_bbox": [10, 10, 100, 100],
                "image_center": [55, 55],
                "image_area": 5000.0,
            },
        ],
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


class LeakTextTranslator(MockIslandTranslator):
    """Always returns a prompt-leaking / garbage output."""

    def __init__(self, text: str = "你是日文漫画翻译。只输出译文。", **kwargs):
        super().__init__(**kwargs)
        self._text = text

    def translate(self, request, glossary):
        self.requests.append(request)
        return TranslationResponse(text=self._text, backend=self.name, model=self.model)


class ValidationGuardTests(unittest.TestCase):
    def test_rejects_new_leak_markers(self):
        with self.assertRaises(TranslationResponseError):
            _validate_island_translation("你是日文漫画翻译。把这段翻成简体中文。", "こんにちは")

    def test_rejects_repeated_single_line(self):
        text = "\n".join(["你：『…』"] * 5)
        with self.assertRaises(TranslationResponseError):
            _validate_island_translation(text, "ああ")

    def test_allows_legit_multi_line(self):
        text = "第一行\n第二行\n第三行"
        _validate_island_translation(text, "長いセリフ")

    def test_short_symbol_detection(self):
        self.assertTrue(_is_short_symbol("！？"))
        self.assertTrue(_is_short_symbol("♡"))
        self.assertFalse(_is_short_symbol("こんにちは"))
        self.assertFalse(_is_short_symbol("あ"))


class EngineAttemptGuardTests(unittest.TestCase):
    def test_short_symbol_never_calls_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset()
            translator = LeakTextTranslator()
            result = translate_island_page(
                dataset,
                {"island_001": "！？"},
                IslandTranslateConfig(precision="local"),
                cache=_fresh_cache(tmp),
                translator_factory=lambda config, precision, style: translator,
            )
            unit = result.units[0]
            self.assertEqual(unit.status, "kept-source")
            self.assertEqual(unit.translated_text, "！？")
            self.assertEqual(len(translator.requests), 0)
            self.assertIn("short-symbol", [a[1] for a in unit.attempts])

    def test_caps_engine_attempts_at_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset()
            translator = LeakTextTranslator()
            result = translate_island_page(
                dataset,
                {"island_001": "これは長い日本語のセリフです"},
                IslandTranslateConfig(
                    precision="online",
                    online_base_url="http://127.0.0.1:9999/v1",
                    online_model="test",
                ),
                cache=_fresh_cache(tmp),
                translator_factory=lambda config, precision, style: translator,
            )
            unit = result.units[0]
            self.assertEqual(unit.status, "kept-source")
            self.assertEqual(unit.translated_text, "これは長い日本語のセリフです")
            # online contextual(1) + online concise(2) + local contextual(3)
            self.assertEqual(len(translator.requests), 3)

    def test_kept_source_logged(self):
        import logging

        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset()
            translator = LeakTextTranslator()
            with self.assertLogs(
                "manga_localizer.island_translate", level=logging.WARNING
            ) as captured:
                translate_island_page(
                    dataset,
                    {"island_001": "これは長い日本語のセリフです"},
                    IslandTranslateConfig(precision="local"),
                    cache=_fresh_cache(tmp),
                    translator_factory=lambda config, precision, style: translator,
                )
            self.assertTrue(
                any("kept-source" in line for line in captured.output)
            )


if __name__ == "__main__":
    unittest.main()
