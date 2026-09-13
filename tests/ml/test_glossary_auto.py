"""拟声词自动进术语库（manga_localizer.glossary_auto）的测试。

三道护栏各自成测：原文必须是短拟声、译文必须短且不是疑问词、译法必须
一致且重复出现；另外验证手写条目不被覆盖、冲突时自动撤出、以及收集
（collect_observations）只看通过护栏的配对。
"""

import json
import tempfile
import unittest
from pathlib import Path

from manga_localizer.glossary_auto import (
    AUTO_NOTE_PREFIX,
    collect_observations,
    sync_sfx_terms,
)


class CollectObservationsTests(unittest.TestCase):
    def test_sfx_pair_collected(self):
        texts = {"1": {"island_001": "ぎゅっ♡"}}
        translations = {"1": {"island_001": "夹紧♡"}}
        self.assertEqual(
            collect_observations(texts, translations),
            {"ぎゅっ♡": {"夹紧♡": 1}},
        )

    def test_counts_accumulate_across_pages(self):
        texts = {"1": {"a": "ぎゅっ♡"}, "2": {"b": "ぎゅっ♡"}}
        translations = {"1": {"a": "夹紧♡"}, "2": {"b": "夹紧♡"}}
        self.assertEqual(
            collect_observations(texts, translations)["ぎゅっ♡"], {"夹紧♡": 2},
        )

    def test_dialogue_with_kanji_not_collected(self):
        texts = {"1": {"a": "これはテスト用の長文"}}
        translations = {"1": {"a": "今天也照常"}}
        self.assertEqual(collect_observations(texts, translations), {})

    def test_suspect_question_target_not_collected(self):
        """ぎゅっ♡ 被读成 どうし 再译成"为什么"——这种绝不许进库。"""
        texts = {"1": {"a": "どうし"}}
        translations = {"1": {"a": "为什么"}}
        self.assertEqual(collect_observations(texts, translations), {})

    def test_long_target_not_collected(self):
        texts = {"1": {"a": "ぎゅっ♡"}}
        translations = {"1": {"a": "紧紧地抱住并且夹紧"}}
        self.assertEqual(collect_observations(texts, translations), {})

    def test_multiline_target_not_collected(self):
        texts = {"1": {"a": "ぎゅっ♡"}}
        translations = {"1": {"a": "夹紧\n再夹"}}
        self.assertEqual(collect_observations(texts, translations), {})

    def test_missing_translation_skipped(self):
        texts = {"1": {"a": "ぎゅっ♡", "b": "ちゅっ"}}
        translations = {"1": {"a": "夹紧♡"}}
        self.assertEqual(
            collect_observations(texts, translations),
            {"ぎゅっ♡": {"夹紧♡": 1}},
        )


class SyncSfxTermsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.path = self.directory / "glossary.json"
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "terms": {"ぎゅっ♡": {"translation": "夹紧♡", "note": "用户手写"}},
                    "names": {"ワテクア": {"translation": "瓦特库娅", "note": ""}},
                    "notes": ["保持口语"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _terms(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_stable_new_sfx_added(self):
        report = sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 3}})
        self.assertEqual(report["added"], ["ちゅっ♡"])
        entry = self._terms()["terms"]["ちゅっ♡"]
        self.assertEqual(entry["translation"], "啾♡")
        self.assertTrue(entry["note"].startswith(AUTO_NOTE_PREFIX))

    def test_single_occurrence_pending_not_added(self):
        report = sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 1}})
        self.assertEqual(report["added"], [])
        self.assertEqual(report["pending"], ["ちゅっ♡"])
        self.assertNotIn("ちゅっ♡", self._terms()["terms"])

    def test_hand_written_entry_never_overwritten(self):
        report = sync_sfx_terms(self.path, {"ぎゅっ♡": {"夹住♡": 5}})
        self.assertEqual(report["added"], [])
        self.assertEqual(report["updated"], [])
        self.assertEqual(
            self._terms()["terms"]["ぎゅっ♡"]["translation"], "夹紧♡",
        )

    def test_conflict_removes_auto_entry(self):
        sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 3}})
        report = sync_sfx_terms(
            self.path, {"ちゅっ♡": {"啾♡": 2, "亲亲♡": 1}},
        )
        self.assertEqual(report["removed"], ["ちゅっ♡"])
        self.assertNotIn("ちゅっ♡", self._terms()["terms"])

    def test_conflict_keeps_hand_written_entry(self):
        report = sync_sfx_terms(
            self.path, {"ぎゅっ♡": {"夹住♡": 2, "夹紧♡": 1}},
        )
        self.assertEqual(report["removed"], [])
        self.assertIn("ぎゅっ♡", self._terms()["terms"])

    def test_auto_entry_updated_when_now_unanimous(self):
        sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 2}})
        report = sync_sfx_terms(self.path, {"ちゅっ♡": {"亲亲♡": 4}})
        self.assertEqual(report["updated"], ["ちゅっ♡"])
        self.assertEqual(
            self._terms()["terms"]["ちゅっ♡"]["translation"], "亲亲♡",
        )

    def test_same_translation_is_noop(self):
        sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 2}})
        before = self.path.read_text(encoding="utf-8")
        report = sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 9}})
        self.assertEqual(report["added"] + report["updated"] + report["removed"], [])
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_min_count_zero_disables(self):
        report = sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 9}}, min_count=0)
        self.assertEqual(report["added"], [])
        self.assertNotIn("ちゅっ♡", self._terms()["terms"])

    def test_other_fields_untouched(self):
        sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 3}})
        data = self._terms()
        self.assertEqual(data["names"]["ワテクア"]["translation"], "瓦特库娅")
        self.assertEqual(data["notes"], ["保持口语"])
        self.assertEqual(data["version"], 1)

    def test_missing_file_creates_terms_section(self):
        path = self.directory / "new.json"
        report = sync_sfx_terms(path, {"ちゅっ♡": {"啾♡": 2}})
        self.assertEqual(report["added"], ["ちゅっ♡"])
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["terms"]["ちゅっ♡"]["translation"],
            "啾♡",
        )

    def test_glossary_still_loadable_by_pipeline(self):
        """入库后术语库必须仍能被 load_glossary 解析（sha 变化 → 缓存失效）。"""
        from manga_localizer.translator import load_glossary

        before = load_glossary(self.path).sha256
        sync_sfx_terms(self.path, {"ちゅっ♡": {"啾♡": 3}})
        after = load_glossary(self.path)
        self.assertNotEqual(before, after.sha256)
        self.assertIn("ちゅっ♡ -> 啾♡", after.prompt_text("ちゅっ♡"))


if __name__ == "__main__":
    unittest.main()
