from __future__ import annotations

import json
import unittest
from pathlib import Path

from manga_localizer.errors import TranslationError
from manga_localizer.island_contract import dataset_from_data
from manga_localizer.island_translate import (
    IslandTranslateConfig,
    MockIslandTranslator,
    PageBatchCache,
    build_island_request,
    translate_island_page,
    translate_page_batch,
    _parse_batch_response,
)
from manga_localizer.translator import TranslationCache


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
            {
                "id": "island_002",
                "image_bbox": [200, 10, 300, 100],
                "image_center": [250, 55],
                "image_area": 5000.0,
            },
        ],
    }
    return dataset_from_data(data, path=__file__, source_sha256="a" * 64)


def _mock_factory(config: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"):
    return MockIslandTranslator(
        {
            "island_001": "模拟：我的军队。",
            "island_002": "模拟：哪个国家都。",
        }
    )


class IslandTranslateTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()
        self.texts = {
            "island_001": "自分の軍だけでも",
            "island_002": "どの国も",
        }
        self.config = IslandTranslateConfig(precision="local")

    def test_mock_translates_all(self):
        result = translate_island_page(
            self.dataset,
            self.texts,
            self.config,
            translator_factory=_mock_factory,
        )
        self.assertEqual(result.precision, "local")
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(
            result.translations["island_001"], "模拟：我的军队。"
        )
        self.assertIsNotNone(result.units[0].cache_key)

    def test_cache_second_run_hits(self):
        cache = TranslationCache(Path(__file__).resolve().parent / "tmp_island_cache")
        try:
            first = translate_island_page(
                self.dataset, self.texts, self.config, cache=cache,
                translator_factory=_mock_factory,
            )
            self.assertEqual(first.cache_hit_count, 0)
            second = translate_island_page(
                self.dataset, self.texts, self.config, cache=cache,
                translator_factory=_mock_factory,
            )
            self.assertEqual(second.cache_hit_count, 2)
            self.assertTrue(all(unit.cache_hit for unit in second.units))
        finally:
            for path in cache.directory.glob("*.json"):
                path.unlink(missing_ok=True)
            cache.directory.rmdir()

    def test_cache_binds_precision(self):
        cache = TranslationCache(Path(__file__).resolve().parent / "tmp_island_cache")

        def precision_factory(config: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"):
            return MockIslandTranslator(
                {"island_001": f"[{precision}] 译文"},
                prefix=f"[{precision}] ",
            )

        try:
            local = IslandTranslateConfig(precision="local")
            online = IslandTranslateConfig(
                precision="online", online_base_url="http://x/v1", online_model="gpt"
            )
            translate_island_page(
                self.dataset, self.texts, local, cache=cache,
                translator_factory=precision_factory,
            )
            online_result = translate_island_page(
                self.dataset, self.texts, online, cache=cache,
                translator_factory=precision_factory,
            )
            self.assertEqual(online_result.cache_hit_count, 0)
        finally:
            for path in cache.directory.glob("*.json"):
                path.unlink(missing_ok=True)
            cache.directory.rmdir()

    def test_online_failure_falls_back_to_local(self):
        seen = []
        cache = TranslationCache(Path(__file__).resolve().parent / "tmp_island_cache2")

        def factory(config: IslandTranslateConfig, precision: str, prompt_style: str = "contextual"):
            if precision == "online":
                mock = MockIslandTranslator(fail_islands=("island_001",))
            else:
                mock = MockIslandTranslator({"island_001": "本地译文"})
            seen.append(precision)
            return mock

        config = IslandTranslateConfig(
            precision="online", online_base_url="http://x/v1", online_model="gpt"
        )
        try:
            result = translate_island_page(
                self.dataset,
                {"island_001": "自分の軍だけでも"},
                config,
                cache=cache,
                translator_factory=factory,
            )
        finally:
            for path in cache.directory.glob("*.json"):
                path.unlink(missing_ok=True)
            cache.directory.rmdir()
        self.assertEqual(seen, ["online", "online", "local"])
        unit = result.units[0]
        self.assertEqual(unit.status, "translated")
        self.assertEqual(unit.translated_text, "本地译文")
        self.assertEqual(unit.backend, "mock-island")
        self.assertTrue(unit.attempts[0][0] == "online" and unit.attempts[0][1].startswith("failed"))
        self.assertIn(("local", "translated"), unit.attempts)

    def test_rejects_identical_output_keeps_source(self):
        def factory(config, precision, prompt_style="contextual"):
            return MockIslandTranslator({"island_001": "自分の軍だけでも"})

        result = translate_island_page(
            self.dataset,
            {"island_001": "自分の軍だけでも"},
            self.config,
            translator_factory=factory,
        )
        unit = result.units[0]
        self.assertEqual(unit.status, "kept-source")
        self.assertEqual(unit.translated_text, "自分の軍だけでも")
        self.assertIn("unchanged source", unit.failure_reason)

    def test_missing_online_config_rejected(self):
        with self.assertRaises(TranslationError):
            translate_island_page(
                self.dataset,
                self.texts,
                IslandTranslateConfig(precision="online"),
                translator_factory=_mock_factory,
            )

    def test_request_binding(self):
        request = build_island_request(
            self.dataset,
            "island_001",
            "自分の軍だけでも",
            context_before=[("island_002", "どの国も")],
        )
        self.assertEqual(request.unit_id, "island_001")
        self.assertEqual(request.ocr_effective_sha256, self.dataset.effective_sha256)
        self.assertEqual(request.translation_query, "自分の軍だけでも")


class PageBatchTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()
        self.texts = {"island_001": "自分の軍だけでも", "island_002": "どの国も"}
        self.config = IslandTranslateConfig(
            precision="online", online_base_url="http://x/v1", online_model="gpt"
        )

    def test_parse_batch_response(self):
        numbered = {1: "island_001", 2: "island_002"}
        translations, failures, errors = _parse_batch_response(
            "1: 我的军队\n2: 哪个国家\n", numbered
        )
        self.assertEqual(translations, {"island_001": "我的军队", "island_002": "哪个国家"})
        self.assertEqual(failures, ())
        self.assertEqual(errors, ())

    def test_parse_missing_and_duplicate(self):
        numbered = {1: "island_001", 2: "island_002", 3: "island_003"}
        translations, failures, errors = _parse_batch_response(
            "1: 甲\n1: 甲重复\n3: 丙\n", numbered
        )
        self.assertEqual(translations, {"island_001": "甲", "island_003": "丙"})
        self.assertEqual(failures, ("island_002",))
        self.assertTrue(any("duplicate" in error for error in errors))

    def test_parse_same_line_numbered_entries(self):
        # DeepSeek often emits every numbered entry space-separated on ONE
        # line; the parser must not anchor to line boundaries.
        numbered = {1: "island_001", 2: "island_002", 3: "island_003"}
        translations, failures, errors = _parse_batch_response(
            "1: 喂喂 2: 手松开了 3: 差不多了结了吧？", numbered
        )
        self.assertEqual(
            translations,
            {"island_001": "喂喂", "island_002": "手松开了", "island_003": "差不多了结了吧？"},
        )
        self.assertEqual(failures, ())
        self.assertEqual(errors, ())

    def test_batch_translate_and_cache(self):
        cache_dir = Path(__file__).resolve().parent / "tmp_batch_cache"
        cache = PageBatchCache(cache_dir)
        calls = []

        def chat_fn(messages):
            calls.append(messages)
            body = messages[1]["content"]
            lines = [line for line in body.splitlines() if line.strip()]
            return "\n".join(f"{line.split(':', 1)[0]}: 译{line.split(':', 1)[0]}" for line in lines)

        try:
            first = translate_page_batch(
                self.dataset, self.texts, self.config, cache=cache, chat_fn=chat_fn
            )
            self.assertFalse(first.cache_hit)
            self.assertEqual(len(first.translations), 2)
            self.assertEqual(first.failures, ())
            second = translate_page_batch(
                self.dataset, self.texts, self.config, cache=cache, chat_fn=chat_fn
            )
            self.assertTrue(second.cache_hit)
            self.assertEqual(second.translations, first.translations)
            self.assertEqual(len(calls), 1)  # second run served from cache
        finally:
            for path in cache.directory.glob("*.json"):
                path.unlink(missing_ok=True)
            cache.directory.rmdir()

    def test_batch_requires_online(self):
        with self.assertRaises(TranslationError):
            translate_page_batch(
                self.dataset,
                self.texts,
                IslandTranslateConfig(precision="local"),
            )


class BatchCacheGlossaryTests(unittest.TestCase):
    """术语表进页级批量缓存签名。

    术语是提示词的一部分（拟声词自动收录会改它）。不进签名就会出现
    "改了术语表却命中旧缓存、改动毫无效果"——全书统一译法也就无从生效。
    """

    def setUp(self):
        self.dataset = _dataset()
        self.texts = {"island_001": "ぎゅっ♡", "island_002": "どの国も"}
        self.config = IslandTranslateConfig(
            precision="online",
            online_base_url="https://api.example/v1",
            online_model="deepseek-flash",
        )
        self.cache = PageBatchCache(
            Path(__file__).resolve().parent / "tmp_batch_glossary_cache"
        )

    def _write_glossary(self, terms):
        import json
        import tempfile

        path = Path(tempfile.mkdtemp()) / "glossary.json"
        path.write_text(
            json.dumps({"version": 1, "terms": terms}, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_glossary_change_changes_signature(self):
        from manga_localizer.translator import load_glossary

        base = load_glossary(self._write_glossary({}))
        with_term = load_glossary(
            self._write_glossary({"ぎゅっ♡": {"translation": "夹紧♡"}})
        )
        first = self.cache.signature(self.dataset, self.texts, self.config, base)
        second = self.cache.signature(self.dataset, self.texts, self.config, with_term)
        self.assertNotEqual(first, second)

    def test_same_glossary_same_signature(self):
        from manga_localizer.translator import load_glossary

        base = load_glossary(self._write_glossary({}))
        self.assertEqual(
            self.cache.signature(self.dataset, self.texts, self.config, base),
            self.cache.signature(self.dataset, self.texts, self.config, base),
        )

    def test_glossary_change_busts_cache(self):
        from manga_localizer.translator import load_glossary

        path = self._write_glossary({})
        calls = []

        def chat_fn(messages):
            calls.append(messages)
            return "1: 夹紧♡\n2: 哪个国家"

        try:
            translate_page_batch(
                self.dataset, self.texts, self.config,
                glossary=load_glossary(path), cache=self.cache, chat_fn=chat_fn,
            )
            self.assertEqual(len(calls), 1)
            # 术语表变了 → 旧译文是按旧提示词产生的，必须重新请求
            path.write_text(
                json.dumps(
                    {"version": 1, "terms": {"ぎゅっ♡": {"translation": "夹紧♡"}}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            second = translate_page_batch(
                self.dataset, self.texts, self.config,
                glossary=load_glossary(path), cache=self.cache, chat_fn=chat_fn,
            )
            self.assertFalse(second.cache_hit)
            self.assertEqual(len(calls), 2)
        finally:
            for stale in self.cache.directory.glob("*.json"):
                stale.unlink(missing_ok=True)
            self.cache.directory.rmdir()


if __name__ == "__main__":
    unittest.main()
