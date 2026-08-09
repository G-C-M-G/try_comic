"""OCR engine selection (--ocr-engine / --stage2-engine) tests for
one_click_gui: combo defaults, settings persistence and _base_args wiring."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from one_click_gui import MainWindow


class OcrEngineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _make_window(self, tmp: Path):
        settings_path = tmp / "api_keys.json"

        class Window(MainWindow):
            @staticmethod
            def _theme_path():
                return tmp / "ui.json"

            @staticmethod
            def _settings_path():
                return settings_path

        return Window()

    def test_default_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._make_window(Path(tmp))
            try:
                self.assertEqual(window.ocr_engine_combo.currentData(), "auto")
                self.assertEqual(window.stage2_engine_combo.currentData(), "manga-ocr")
            finally:
                window.close()

    def test_save_settings_writes_engine_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                index = window.ocr_engine_combo.findData("paddle")
                self.assertGreaterEqual(index, 0)
                window.ocr_engine_combo.setCurrentIndex(index)
                index = window.stage2_engine_combo.findData("ai-ocr")
                self.assertGreaterEqual(index, 0)
                window.stage2_engine_combo.setCurrentIndex(index)
                window._save_settings()
                data = json.loads(
                    (tmp_path / "api_keys.json").read_text(encoding="utf-8"))
                self.assertEqual(data["ocr_engine"], "paddle")
                self.assertEqual(data["stage2_engine"], "ai-ocr")
            finally:
                window.close()

    def test_load_settings_restores_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                (tmp_path / "api_keys.json").write_text(
                    json.dumps({"ocr_engine": "windows-ocr",
                                "stage2_engine": "ai-ocr"}),
                    encoding="utf-8",
                )
                window._load_settings()
                self.assertEqual(window.ocr_engine_combo.currentData(), "windows-ocr")
                self.assertEqual(window.stage2_engine_combo.currentData(), "ai-ocr")
            finally:
                window.close()

    def test_load_settings_invalid_value_keeps_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                (tmp_path / "api_keys.json").write_text(
                    json.dumps({"ocr_engine": "bogus", "stage2_engine": "bogus"}),
                    encoding="utf-8",
                )
                window._load_settings()
                self.assertEqual(window.ocr_engine_combo.currentData(), "auto")
                self.assertEqual(window.stage2_engine_combo.currentData(), "manga-ocr")
            finally:
                window.close()

    def test_base_args_default_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                window.images_edit.setText(str(tmp_path))
                args = window._base_args()
                self.assertNotIn("--ocr-engine", args)
                self.assertNotIn("--stage2-engine", args)
            finally:
                window.close()

    def test_base_args_non_default_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                window.images_edit.setText(str(tmp_path))
                index = window.ocr_engine_combo.findData("paddle")
                window.ocr_engine_combo.setCurrentIndex(index)
                index = window.stage2_engine_combo.findData("ai-ocr")
                window.stage2_engine_combo.setCurrentIndex(index)
                args = window._base_args()
                self.assertIn("--ocr-engine", args)
                self.assertEqual(args[args.index("--ocr-engine") + 1], "paddle")
                self.assertIn("--stage2-engine", args)
                self.assertEqual(
                    args[args.index("--stage2-engine") + 1], "ai-ocr")
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
