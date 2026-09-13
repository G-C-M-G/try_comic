"""OCR engine selection (--ocr-engine / --stage2-engine) tests.

Covers: forced capture-stage engine passthrough in ``stage1_capture``,
forced ai-ocr stage2 validation/dispatch in ``run_manga`` (all boxes to the
vision model, failures falling back to manga-ocr), the unchanged default
manga-ocr dispatch, and the CLI wiring of both flags through ``main()``
into the stage functions.  Everything is mocked: no real manga-ocr
subprocess and no network.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from one_click_translate import main, run_manga


def _crop(crop_id, bbox=(0, 0, 10, 10)):
    return {"id": crop_id, "source": "x.jpg", "bbox": list(bbox)}


class RunMangaAiOcrTests(unittest.TestCase):
    """stage2-engine=ai-ocr: forced validation and AI-only reading."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.crops = [_crop("1:box:a"), _crop("1:box:b")]

    def test_ai_ocr_forced_without_config_raises(self):
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=self.crops):
            with self.assertRaisesRegex(
                RuntimeError, "stage2-engine ai-ocr"
            ):
                run_manga(self.crops, self.root, stage2_engine="ai-ocr")

    def test_ai_ocr_forced_configured_reads_all_misses(self):
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=self.crops), \
             mock.patch("one_click_translate._store_readings_cache"), \
             mock.patch("manga_localizer.ai_ocr.ai_ocr_readings",
                        return_value={"1:box:a": "あ", "1:box:b": "い"}) as ai, \
             mock.patch("one_click_translate._run_manga_subprocess") as sub:
            result, fallbacks = run_manga(
                self.crops, self.root,
                ai_ocr_url="https://vision.example",
                ai_ocr_model="glm-4v",
                ai_ocr_api_key="secret",
                stage2_engine="ai-ocr",
            )
            self.assertEqual(result, {"1:box:a": "あ", "1:box:b": "い"})
            self.assertEqual(fallbacks, set())
            self.assertEqual(ai.call_args.args[0], self.crops)
            self.assertEqual(
                ai.call_args.args[1:],
                ("https://vision.example", "glm-4v", "secret"),
            )
            sub.assert_not_called()

    def test_ai_ocr_failures_fall_back_to_manga_ocr(self):
        """AI OCR 读空的框交给 manga-ocr（不是 Sakura），并记入 fallback_ids。"""
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=self.crops), \
             mock.patch("one_click_translate._store_readings_cache"), \
             mock.patch("manga_localizer.ai_ocr.ai_ocr_readings",
                        return_value={"1:box:a": "あ"}), \
             mock.patch("one_click_translate._run_manga_subprocess",
                        return_value={"1:box:b": "ぎゅっ"}) as sub:
            result, fallbacks = run_manga(
                self.crops, self.root,
                ai_ocr_url="https://vision.example",
                ai_ocr_model="glm-4v",
                ai_ocr_api_key="secret",
                stage2_engine="ai-ocr",
            )
            self.assertEqual(result, {"1:box:a": "あ", "1:box:b": "ぎゅっ"})
            self.assertEqual(fallbacks, {"1:box:b"})
            self.assertEqual([c["id"] for c in sub.call_args.args[0]],
                             ["1:box:b"])

    def test_ai_ocr_forced_cache_hit_reads_nothing(self):
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={"1:box:a": "あ"}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=[]), \
             mock.patch("one_click_translate._store_readings_cache") as store, \
             mock.patch("manga_localizer.ai_ocr.ai_ocr_readings") as ai, \
             mock.patch("one_click_translate._run_manga_subprocess") as sub:
            result, fallbacks = run_manga(
                self.crops, self.root,
                ai_ocr_url="https://vision.example",
                ai_ocr_model="glm-4v",
                ai_ocr_api_key="secret",
                stage2_engine="ai-ocr",
            )
            self.assertEqual(result, {"1:box:a": "あ", "1:box:b": ""})
            self.assertEqual(fallbacks, set())
            ai.assert_not_called()
            sub.assert_not_called()
            store.assert_called_once_with(self.root, {})


class RunMangaMangaOcrTests(unittest.TestCase):
    """Default stage2-engine=manga-ocr keeps the existing dispatch."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.crops = [_crop("1:box:a")]

    def test_default_no_ai_uses_subprocess(self):
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=self.crops), \
             mock.patch("one_click_translate._store_readings_cache"), \
             mock.patch("manga_localizer.ai_ocr.ai_ocr_readings") as ai, \
             mock.patch("one_click_translate._run_manga_subprocess",
                        return_value={"1:box:a": "あ"}) as sub:
            result, fallbacks = run_manga(self.crops, self.root)
            self.assertEqual(result, {"1:box:a": "あ"})
            self.assertEqual(fallbacks, set())
            sub.assert_called_once()
            ai.assert_not_called()

    def test_default_with_ai_config_still_prefers_ai(self):
        with mock.patch("one_click_translate._load_readings_cache",
                        return_value={}), \
             mock.patch("one_click_translate._select_crop_misses",
                        return_value=self.crops), \
             mock.patch("one_click_translate._store_readings_cache"), \
             mock.patch("manga_localizer.ai_ocr.ai_ocr_readings",
                        return_value={"1:box:a": "あ"}) as ai, \
             mock.patch("one_click_translate._run_manga_subprocess") as sub:
            result, _fallbacks = run_manga(
                self.crops, self.root,
                ai_ocr_url="https://vision.example",
                ai_ocr_model="glm-4v",
                ai_ocr_api_key="secret",
            )
            self.assertEqual(result, {"1:box:a": "あ"})
            ai.assert_called_once()
            sub.assert_not_called()


class SuspiciousFallbackTests(unittest.TestCase):
    """回退读数可疑 → 判定为假选框（不翻译、保留原文）。"""

    def test_suspicious_fallback_dropped(self):
        from one_click_translate import _demote_suspicious_fallbacks

        progress = mock.Mock()
        readings = {"1:box:a": "ぎゅっ♡", "1:box:b": "どうし"}
        dropped = {"1": set()}
        out = _demote_suspicious_fallbacks(
            readings, dropped, {"1:box:a", "1:box:b"}, progress,
        )
        self.assertEqual(out["1"], {"a", "b"})
        self.assertIn("保留原文", progress.log.call_args.args[0])

    def test_normal_fallback_kept(self):
        """回退读数正常（含汉字/长句）时照常翻译。"""
        from one_click_translate import _demote_suspicious_fallbacks

        progress = mock.Mock()
        readings = {"1:box:a": "これはテスト用の長文"}
        dropped = {"1": set()}
        out = _demote_suspicious_fallbacks(
            readings, dropped, {"1:box:a"}, progress,
        )
        self.assertEqual(out["1"], set())
        progress.log.assert_not_called()

    def test_non_fallback_readings_never_demoted(self):
        """只有回退来的读数才走可疑规则（正常 manga-ocr 短假名不丢）。"""
        from one_click_translate import _demote_suspicious_fallbacks

        progress = mock.Mock()
        readings = {"1:box:a": "どうして"}
        dropped = {"1": set()}
        out = _demote_suspicious_fallbacks(readings, dropped, set(), progress)
        self.assertEqual(out["1"], set())
        progress.log.assert_not_called()

    def test_reread_suffix_id_stripped(self):
        from one_click_translate import _demote_suspicious_fallbacks

        progress = mock.Mock()
        readings = {"1:box:a": "ぎゅっ♡"}
        dropped = {"1": set()}
        out = _demote_suspicious_fallbacks(
            readings, dropped, {"1:box:a:reread"}, progress,
        )
        self.assertEqual(out["1"], {"a"})

    def test_already_dropped_not_double_counted(self):
        from one_click_translate import _demote_suspicious_fallbacks

        progress = mock.Mock()
        readings = {"1:box:a": "ぎゅっ♡"}
        dropped = {"1": {"a"}}
        out = _demote_suspicious_fallbacks(
            readings, dropped, {"1:box:a"}, progress,
        )
        self.assertEqual(out["1"], {"a"})
        progress.log.assert_not_called()


class _FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class _FakeProc:
    def __init__(self, lines=()):
        self.stdout = _FakeStream(lines)

    def wait(self):
        return 0


class Stage1EnginePassthroughTests(unittest.TestCase):
    """stage1_capture forwards a forced ocr_engine to photos_capture."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.images = Path(tempfile.mkdtemp())

    def _run(self, ocr_engine):
        from one_click_translate import stage1_capture

        with mock.patch("one_click_translate._purge_page_artifacts"), \
             mock.patch("one_click_translate.subprocess.Popen") as popen:
            calls = []
            popen.side_effect = lambda *a, **k: (calls.append(a[0]), _FakeProc())[1]
            ok = stage1_capture(
                self.images, self.root, {"1"}, 3, 25, None, 0,
                overwrite=True, ocr_engine=ocr_engine,
                box_source="photos",
            )
            self.assertTrue(ok)
            return calls[0]

    def test_forced_engine_passed_through(self):
        cmd = self._run("paddle")
        self.assertIn("--ocr-engine", cmd)
        self.assertEqual(cmd[cmd.index("--ocr-engine") + 1], "paddle")

    def test_auto_omits_flag(self):
        cmd = self._run("auto")
        self.assertNotIn("--ocr-engine", cmd)


class MainFlagWiringTests(unittest.TestCase):
    """--ocr-engine / --stage2-engine reach the stage functions."""

    def setUp(self):
        self.images = Path(tempfile.mkdtemp())
        (self.images / "1.png").write_bytes(b"x")
        (self.images / "2.png").write_bytes(b"x")

    def test_ocr_engine_and_stage2_engine_wired(self):
        with mock.patch("one_click_translate.stage1_capture",
                        return_value=True) as cap, \
             mock.patch("one_click_translate.stage2_process",
                        return_value=True) as proc, \
             mock.patch("one_click_translate._skip_pages",
                        return_value=set()), \
             mock.patch("one_click_translate._review_required_pages",
                        return_value=[]), \
             mock.patch("one_click_translate._load_api_config",
                        return_value={}):
            code = main([
                "--images", str(self.images),
                "--auto",
                "--ocr-engine", "paddle",
                "--stage2-engine", "ai-ocr",
            ])
            self.assertEqual(code, 0)
            self.assertEqual(cap.call_args.kwargs["ocr_engine"], "paddle")
            self.assertEqual(proc.call_args.kwargs["stage2_engine"], "ai-ocr")

    def test_defaults_are_auto_and_manga_ocr(self):
        with mock.patch("one_click_translate.stage1_capture",
                        return_value=True) as cap, \
             mock.patch("one_click_translate.stage2_process",
                        return_value=True) as proc, \
             mock.patch("one_click_translate._skip_pages",
                        return_value=set()), \
             mock.patch("one_click_translate._review_required_pages",
                        return_value=[]), \
             mock.patch("one_click_translate._load_api_config",
                        return_value={}):
            code = main(["--images", str(self.images), "--auto"])
            self.assertEqual(code, 0)
            self.assertEqual(cap.call_args.kwargs["ocr_engine"], "auto")
            self.assertEqual(proc.call_args.kwargs["stage2_engine"], "manga-ocr")

    def test_invalid_engine_choice_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--images", str(self.images), "--ocr-engine", "bogus"])
        self.assertEqual(ctx.exception.code, 2)
        with self.assertRaises(SystemExit) as ctx:
            main(["--images", str(self.images), "--stage2-engine", "bogus"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
