"""Stage-split (--stage) and overwrite (--overwrite) behaviour tests.

Covers: cache-key purging, translations.json round-trip, page-text
rebuild from the OCR cache (dropped-box derivation), stage gating in
``_run_stage2_single``, the full ``stage2_process`` composition, CLI
mutual-exclusion rules, and the stage-1 capture overwrite flag.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from one_click_translate import (
    _load_page_texts,
    _load_translations,
    _purge_page_artifacts,
    _store_translations,
    main,
)


def _box(box_id, x1, y1, x2, y2):
    return {"id": box_id, "image_bbox": [x1, y1, x2, y2], "page_scale": False}


def _make_dataset(out_root, page, islands, blocks, groups):
    page_dir = out_root / "dataset" / page
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "islands.json").write_text(
        json.dumps({"islands": islands}), encoding="utf-8"
    )
    (page_dir / "selection_blocks.json").write_text(
        json.dumps({"selection_blocks": blocks, "selection_groups": groups}),
        encoding="utf-8",
    )


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _stage2_kwargs(**overrides):
    kwargs = dict(
        ai_url="https://api.deepseek.com/v1",
        ai_model="deepseek-chat",
        ai_api_key="",
        ai_ocr_url="",
        ai_ocr_model="",
        ai_ocr_api_key="",
        resume=False,
        no_pdf=True,
        min_font_size=None,
        font_name="",
        stroke_divisor=None,
        row_gap_divisor=None,
        column_gap_divisor=None,
        tracking_divisor=None,
        text_color="",
        stroke_color="",
        background_color="",
        max_workers=1,
    )
    kwargs.update(overrides)
    return kwargs


class PurgeArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "capture" / "photos" / "1").mkdir(parents=True)
        (self.root / "capture" / "photos" / "1" / "capture.json").write_text("{}")
        (self.root / "capture" / "photos" / "2").mkdir(parents=True)
        (self.root / "capture" / "photos" / "2" / "capture.json").write_text("{}")
        (self.root / "check" / "photos").mkdir(parents=True)
        (self.root / "check" / "photos" / "1.bluebox_check.png").write_bytes(b"x")
        (self.root / "check" / "photos" / "2.bluebox_check.png").write_bytes(b"x")
        (self.root / "dataset" / "1").mkdir(parents=True)
        (self.root / "dataset" / "1" / "islands.json").write_text("{}")
        _write_json(self.root / "ocr_readings.json", {
            "1:box:a": "あ", "1:box:b": "い", "2:box:c": "う",
        })
        _write_json(self.root / "translations.json", {
            "1": {"island_001": "甲"}, "2": {"island_001": "乙"},
        })

    def test_purge_everything_for_selected_pages_only(self):
        _purge_page_artifacts(
            self.root, ["1"], capture=True, dataset=True,
            readings=True, translations=True,
        )
        self.assertFalse((self.root / "capture" / "photos" / "1").exists())
        self.assertFalse((self.root / "check" / "photos" / "1.bluebox_check.png").exists())
        self.assertFalse((self.root / "dataset" / "1").exists())
        self.assertTrue((self.root / "capture" / "photos" / "2").exists())
        self.assertTrue((self.root / "check" / "photos" / "2.bluebox_check.png").exists())
        readings = json.loads((self.root / "ocr_readings.json").read_text(encoding="utf-8"))
        self.assertEqual(readings, {"2:box:c": "う"})
        translations = json.loads((self.root / "translations.json").read_text(encoding="utf-8"))
        self.assertEqual(list(translations), ["2"])

    def test_default_purge_touches_only_cache_keys(self):
        _purge_page_artifacts(self.root, ["1"])
        self.assertTrue((self.root / "capture" / "photos" / "1").exists())
        self.assertTrue((self.root / "dataset" / "1").exists())
        readings = json.loads((self.root / "ocr_readings.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(readings), ["2:box:c"])
        translations = json.loads((self.root / "translations.json").read_text(encoding="utf-8"))
        self.assertEqual(list(translations), ["2"])

    def test_purge_missing_page_is_idempotent(self):
        _purge_page_artifacts(self.root, ["99"], capture=True)
        self.assertTrue((self.root / "capture" / "photos" / "1").exists())

    def test_purge_without_cache_files(self):
        (self.root / "ocr_readings.json").unlink()
        (self.root / "translations.json").unlink()
        _purge_page_artifacts(self.root, ["1"])  # must not raise


class TranslationsRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_round_trip(self):
        payload = {"1": {"island_001": "甲", "island_002": "乙"}}
        _store_translations(self.root, payload)
        self.assertEqual(_load_translations(self.root), payload)

    def test_missing_file_yields_empty(self):
        self.assertEqual(_load_translations(self.root), {})

    def test_malformed_file_yields_empty(self):
        (self.root / "translations.json").write_text("not json")
        self.assertEqual(_load_translations(self.root), {})

    def test_non_dict_entries_are_skipped(self):
        _write_json(self.root / "translations.json",
                    {"1": {"island_001": "甲"}, "2": "junk", "3": 5})
        self.assertEqual(_load_translations(self.root), {"1": {"island_001": "甲"}})


class LoadPageTextsTests(unittest.TestCase):
    """Rebuilding texts/directions/dropped from the OCR cache must match
    what the OCR stage derived: a box missing from the cache was read
    empty twice and is dropped."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        islands = [{"id": "island_001", "image_bbox": [0, 0, 1000, 1000]}]
        blocks = [_box("a", 100, 100, 200, 900), _box("b", 300, 100, 400, 900)]
        groups = [{"id": "g1", "members": ["a", "b"]}]
        _make_dataset(self.root, "1", islands, blocks, groups)

    def _readings(self, payload):
        _write_json(self.root / "ocr_readings.json", payload)

    def test_dropped_box_not_in_cache_is_excluded(self):
        self._readings({"1:box:a": "あ"})
        texts, directions, dropped = _load_page_texts(self.root, ["1"])
        self.assertEqual(texts["1"]["island_001"], "あ")
        self.assertEqual(dropped["1"], {"b"})
        self.assertEqual(directions["1"]["island_001"], "vertical")

    def test_vertical_join_right_to_left(self):
        self._readings({"1:box:a": "あ", "1:box:b": "い"})
        texts, _directions, dropped = _load_page_texts(self.root, ["1"])
        self.assertEqual(texts["1"]["island_001"], "いあ")
        self.assertEqual(dropped["1"], set())

    def test_page_without_dataset_is_omitted(self):
        self._readings({"1:box:a": "あ"})
        texts, _directions, dropped = _load_page_texts(self.root, ["1", "99"])
        self.assertIn("1", texts)
        self.assertNotIn("99", texts)
        self.assertNotIn("99", dropped)


class CliArgTests(unittest.TestCase):
    def setUp(self):
        self.images = Path(tempfile.mkdtemp())
        (self.images / "1.jpg").write_bytes(b"x")

    def test_process_and_stage_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--images", str(self.images), "--process", "--stage", "ocr"])
        self.assertEqual(ctx.exception.code, 2)

    def test_stage_value_must_be_valid(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--images", str(self.images), "--stage", "bogus"])
        self.assertEqual(ctx.exception.code, 2)

    def test_overwrite_and_resume_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--images", str(self.images), "--process",
                  "--overwrite", "--resume"])
        self.assertEqual(ctx.exception.code, 2)


class StageGatingTests(unittest.TestCase):
    """_run_stage2_single must pass only the pages whose prerequisites
    exist (OCR readings for translate, translations for typeset)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.images = Path(tempfile.mkdtemp())
        _make_dataset(self.root, "1", [{"id": "island_001", "image_bbox": [0, 0, 10, 10]}],
                      [_box("a", 0, 0, 5, 5)], [{"id": "g1", "members": ["a"]}])
        _make_dataset(self.root, "2", [{"id": "island_001", "image_bbox": [0, 0, 10, 10]}],
                      [_box("a", 0, 0, 5, 5)], [{"id": "g1", "members": ["a"]}])

    def test_translate_only_qualified_pages(self):
        from one_click_translate import _run_stage2_single

        with mock.patch("one_click_translate._load_page_texts") as load_texts, \
             mock.patch("one_click_translate._stage2_translate") as do_translate:
            load_texts.return_value = (
                {"1": {"island_001": "あ"}, "2": {"island_001": ""}}, {}, {})
            do_translate.return_value = {"1": {"island_001": "甲"}}
            ok = _run_stage2_single("translate", self.images, self.root,
                                    ["1", "2"], _stage2_kwargs(), overwrite=False)
            self.assertTrue(ok)
            self.assertEqual(do_translate.call_args.args[1], ["1"])

    def test_typeset_only_translated_pages(self):
        from one_click_translate import _run_stage2_single

        with mock.patch("one_click_translate._load_translations") as load_trans, \
             mock.patch("one_click_translate._load_page_texts") as load_texts, \
             mock.patch("one_click_translate._stage2_typeset") as do_typeset:
            load_trans.return_value = {"1": {"island_001": "甲"}}
            load_texts.return_value = (
                {"1": {"island_001": "あ"}}, {"1": {"island_001": "vertical"}}, {})
            do_typeset.return_value = True
            ok = _run_stage2_single("typeset", self.images, self.root,
                                    ["1", "2"], _stage2_kwargs(), overwrite=False)
            self.assertTrue(ok)
            self.assertEqual(do_typeset.call_args.args[3], ["1"])

    def test_ocr_uses_dataset_stage_then_ocrs(self):
        from one_click_translate import _run_stage2_single

        with mock.patch("one_click_translate._stage2_dataset") as do_dataset, \
             mock.patch("one_click_translate._stage2_ocr") as do_ocr:
            do_dataset.return_value = ["1"]
            ok = _run_stage2_single("ocr", self.images, self.root,
                                    ["1", "2"], _stage2_kwargs(), overwrite=False)
            self.assertTrue(ok)
            self.assertEqual(do_ocr.call_args.args[1], ["1"])


class OrchestratorTests(unittest.TestCase):
    """stage2_process composes the four stage functions in order."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.images = Path(tempfile.mkdtemp())

    def test_full_chain_propagates_results(self):
        from one_click_translate import stage2_process

        with mock.patch("one_click_translate._stage2_dataset") as ds, \
             mock.patch("one_click_translate._stage2_ocr") as ocr, \
             mock.patch("one_click_translate._stage2_translate") as tr, \
             mock.patch("one_click_translate._stage2_typeset") as ts:
            ds.return_value = ["1", "2"]
            ocr.return_value = (
                {"1": {"island_001": "あ"}}, {"1": {"island_001": "vertical"}}, {})
            tr.return_value = {"1": {"island_001": "甲"}}
            ts.return_value = True
            ok = stage2_process(self.images, self.root, ["1", "2"],
                                max_workers=2, no_pdf=True)
            self.assertTrue(ok)
            self.assertEqual(ocr.call_args.args[1], ["1", "2"])
            self.assertEqual(tr.call_args.args[1], ["1", "2"])
            self.assertEqual(tr.call_args.kwargs["max_workers"], 2)
            self.assertEqual(ts.call_args.args[3], ["1", "2"])
            self.assertEqual(ts.call_args.args[4], tr.return_value)

    def test_overwrite_propagates_to_all_stages(self):
        from one_click_translate import stage2_process

        with mock.patch("one_click_translate._stage2_dataset") as ds, \
             mock.patch("one_click_translate._stage2_ocr") as ocr, \
             mock.patch("one_click_translate._stage2_translate") as tr, \
             mock.patch("one_click_translate._stage2_typeset") as ts:
            ds.return_value = ["1"]
            ocr.return_value = ({"1": {}}, {"1": {}}, {})
            tr.return_value = {"1": {}}
            ts.return_value = True
            stage2_process(self.images, self.root, ["1"], overwrite=True)
            self.assertTrue(ds.call_args.kwargs["overwrite"])
            self.assertTrue(ocr.call_args.kwargs["overwrite"])
            self.assertTrue(tr.call_args.kwargs["overwrite"])


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


class Stage1OverwriteTests(unittest.TestCase):
    """stage1_capture: --overwrite purges old artifacts and drops --resume
    so the selected pages are re-captured."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.images = Path(tempfile.mkdtemp())

    def _run(self, overwrite):
        from one_click_translate import stage1_capture

        with mock.patch("one_click_translate._purge_page_artifacts") as purge, \
             mock.patch("one_click_translate.subprocess.Popen") as popen:
            proc = _FakeProc()
            popen.return_value = proc
            calls = []
            popen.side_effect = lambda *a, **k: (calls.append(a[0]), proc)[1]
            ok = stage1_capture(
                self.images, self.root, {"1", "2"}, 3, 25, None, 0,
                overwrite=overwrite, box_source="photos",
            )
            self.assertTrue(ok)
            self.assertEqual(len(calls), 1)
            return calls[0], purge.call_args

    def test_overwrite_purges_and_drops_resume(self):
        cmd, purge = self._run(overwrite=True)
        self.assertNotIn("--resume", cmd)
        self.assertIsNotNone(purge)
        self.assertEqual(purge.args[1], ["1", "2"])
        self.assertTrue(purge.kwargs["capture"])

    def test_no_overwrite_keeps_resume_and_skips_purge(self):
        cmd, purge = self._run(overwrite=False)
        self.assertIn("--resume", cmd)
        self.assertIsNone(purge)


if __name__ == "__main__":
    unittest.main()


class TranslationsStoreMergeTests(unittest.TestCase):
    """translations.json 必须**按页合并**写入。

    回归：原实现是整文件替换，`--pages 1` 这类子集运行会把其余页的译文整本
    抹掉（实测踩到——跑完单页验证后 31 页译文消失，后续嵌字只剩 1 页）。
    """

    def test_subset_store_keeps_other_pages(self):
        from one_click_translate import _load_translations, _store_translations

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store_translations(root, {
                "1": {"island_001": "甲"},
                "2": {"island_001": "乙"},
                "3": {"island_001": "丙"},
            })
            # 只重写第 2 页（模拟 --pages 2）
            _store_translations(root, {"2": {"island_001": "乙改"}})
            saved = _load_translations(root)
            self.assertEqual(sorted(saved), ["1", "2", "3"])
            self.assertEqual(saved["2"]["island_001"], "乙改")
            self.assertEqual(saved["1"]["island_001"], "甲")
            self.assertEqual(saved["3"]["island_001"], "丙")

    def test_page_rewrite_replaces_whole_page(self):
        from one_click_translate import _load_translations, _store_translations

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _store_translations(root, {"1": {"island_001": "甲", "island_002": "乙"}})
            # 该页重译后岛集合变了：整页替换，不留旧岛
            _store_translations(root, {"1": {"island_001": "新甲"}})
            saved = _load_translations(root)
            self.assertEqual(saved["1"], {"island_001": "新甲"})


class PurgeBackupTests(unittest.TestCase):
    """`--overwrite` 清空产物前必须先备份（失败可回滚）。

    背景：`--overwrite` 在 stage2 开头就删掉选中页的读数/译文，而重新生成要跑完
    OCR + 翻译；中途失败就是"只丢不进"（实测 24 页译文被清成 {}）。
    """

    def setUp(self):
        import json
        import tempfile

        self.root = Path(tempfile.mkdtemp())
        (self.root / "translations.json").write_text(
            json.dumps(
                {"1": {"island_001": "甲"}, "2": {"island_002": "乙"}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (self.root / "ocr_readings.json").write_text(
            json.dumps({"1:box:a": "あ", "2:box:b": "い"}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _purge(self, pages):
        from one_click_translate import _purge_page_artifacts

        _purge_page_artifacts(self.root, pages, capture=False, dataset=False,
                              readings=True, translations=True)

    def _load(self, name):
        import json

        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def test_backup_written_before_purge(self):
        self._purge(["1"])
        self.assertEqual(self._load("translations.json"), {"2": {"island_002": "乙"}})
        self.assertEqual(
            self._load("translations.json.bak"),
            {"1": {"island_001": "甲"}, "2": {"island_002": "乙"}},
        )
        self.assertEqual(self._load("ocr_readings.json.bak"),
                         {"1:box:a": "あ", "2:box:b": "い"})

    def test_failed_rerun_keeps_the_backup(self):
        """失败后重跑（文件已空）不能覆盖掉可回滚的那份备份。"""
        self._purge(["1"])
        before = (self.root / "translations.json.bak").read_text(encoding="utf-8")
        # 第二次：这两个键已不存在 → 不该产生新备份
        self._purge(["1"])
        self.assertEqual(
            (self.root / "translations.json.bak").read_text(encoding="utf-8"), before,
        )

    def test_no_backup_when_nothing_to_remove(self):
        self._purge(["999"])
        self.assertFalse((self.root / "translations.json.bak").exists())
        self.assertEqual(len(self._load("translations.json")), 2)

    def test_pages_not_selected_untouched(self):
        self._purge(["2"])
        self.assertEqual(self._load("translations.json"), {"1": {"island_001": "甲"}})


class TypesetPngSaveTests(unittest.TestCase):
    """嵌字 PNG 写盘：原子替换 + 被占用时给出可执行的报错。

    实测踩到：图片查看器开着 `typeset/*.typeset.png`，重嵌字写盘失败，Windows
    报 `[Errno 22] Invalid argument`（对已内存映射文件的写返回
    ERROR_INVALID_PARAMETER），看起来像"文件名非法"，其实是"文件被占用"。
    """

    def setUp(self):
        import tempfile

        from PIL import Image

        self.dir = Path(tempfile.mkdtemp())
        self.image = Image.new("RGB", (8, 8), "white")

    def test_writes_target_and_leaves_no_tmp(self):
        from one_click_translate import _save_typeset_png

        target = self.dir / "1.typeset.png"
        _save_typeset_png(self.image, target)
        self.assertTrue(target.is_file())
        self.assertEqual(list(self.dir.glob("*.tmp*")), [])

    def test_overwrites_existing(self):
        from one_click_translate import _save_typeset_png

        target = self.dir / "1.typeset.png"
        target.write_bytes(b"stale")
        _save_typeset_png(self.image, target)
        self.assertGreater(target.stat().st_size, 6)

    def test_failure_raises_actionable_error_and_cleans_tmp(self):
        from manga_localizer.errors import ErasureError
        from one_click_translate import _save_typeset_png

        # 用一个目录当目标：临时文件写得成，替换必然失败
        target = self.dir / "2.typeset.png"
        target.mkdir()
        with self.assertRaises(ErasureError) as ctx:
            _save_typeset_png(self.image, target)
        self.assertIn("被图片查看器", str(ctx.exception))
        self.assertEqual(list(self.dir.glob("*.tmp*")), [])


class UnfittableIslandSkipTests(unittest.TestCase):
    """放不下的岛保留原图（不擦、不写）。

    盒子 15×40px 却要塞 60 个字：12px 绝对下限也放不下，只能被掩膜裁掉 ——
    成品里是一个看不见的小点，比保留原文更差。实测 丙书 有 17 个这种岛。
    """

    def _dataset(self, bbox, tmp):
        from manga_localizer.island_contract import dataset_from_data

        payload = {
            "schema_version": 1, "page": "1", "image_size": [600, 800],
            "islands": [{
                "id": "island_001", "image_bbox": list(bbox),
                "image_center": [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2],
                "image_area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            }],
        }
        return dataset_from_data(payload, path=Path(tmp) / "islands.json",
                                 source_sha256="t")

    def _probe(self, bbox, text, tmp):
        from manga_localizer.island_typeset import TypesetConfig
        from one_click_translate import _unfittable_island_ids

        dataset = self._dataset(bbox, tmp)
        return _unfittable_island_ids(
            dataset, {"island_001": text}, TypesetConfig(),
            directions={"island_001": "vertical"},
        )

    def test_tiny_box_with_long_text_is_unfittable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._probe((100, 100, 115, 140), "六十字的译文" * 10, tmp),
                {"island_001"},
            )

    def test_normal_box_is_fittable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._probe((100, 100, 200, 400), "正常译文", tmp), set())

    def test_small_box_with_short_text_is_fittable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._probe((100, 100, 140, 260), "四个字", tmp), set())

    def test_empty_translations_is_noop(self):
        from manga_localizer.island_typeset import TypesetConfig
        from one_click_translate import _unfittable_island_ids

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset((100, 100, 115, 140), tmp)
            self.assertEqual(
                _unfittable_island_ids(dataset, {}, TypesetConfig()), set(),
            )
