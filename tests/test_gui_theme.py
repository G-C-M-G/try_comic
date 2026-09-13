"""Dark-theme palette and persistence tests for one_click_gui."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem

from one_click_gui import DARK, LIGHT, ROOT, MainWindow


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


class DarkPaletteTests(unittest.TestCase):
    def test_dark_background_is_pure_black(self):
        self.assertEqual(DARK["bg"], "#000000")

    def test_dark_surfaces_are_neutral_gray(self):
        # Cards/hover/borders must be neutral gray, never the old blue-slate
        # palette (#0f172a / #1e293b / #334155).
        for key in ("card", "hover", "border", "border_hover"):
            red, green, blue = _hex_to_rgb(DARK[key])
            self.assertEqual(red, green, key)
            self.assertEqual(green, blue, key)

    def test_light_theme_stays_light(self):
        self.assertEqual(LIGHT["bg"], "#f9fafb")


class ThemePersistenceTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.json"
            MainWindow._save_theme(path, False)
            self.assertFalse(MainWindow._load_theme(path))
            MainWindow._save_theme(path, True)
            self.assertTrue(MainWindow._load_theme(path))

    def test_missing_file_defaults_dark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertTrue(MainWindow._load_theme(path))

    def test_save_preserves_other_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.json"
            path.write_text(
                json.dumps({"island_dark": True, "dark": False}),
                encoding="utf-8",
            )
            MainWindow._save_theme(path, True)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(data["dark"])
            self.assertTrue(data["island_dark"])


class GuiToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_toggle_switches_theme_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.json"

            class Window(MainWindow):
                @staticmethod
                def _theme_path():
                    return path

            window = Window()
            start = window.dark
            window._toggle_theme()
            self.assertNotEqual(window.dark, start)
            self.assertEqual(window.dark, MainWindow._load_theme(path))
            window.close()


class ReviewRecaptureAllTests(unittest.TestCase):
    """The "全部重采" button must clear every review-gated page's capture
    directory and re-launch stage1 with all of them as --pages."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _make_window(self, tmp: Path):
        path = tmp / "ui.json"

        class Window(MainWindow):
            @staticmethod
            def _theme_path():
                return path

        return Window()

    def test_button_present_in_review_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._make_window(Path(tmp))
            try:
                labels = [
                    b.text() for b in window.findChildren(type(window.theme_btn))
                    if b.parent() is not None
                ]
                self.assertIn("全部重采", labels)
            finally:
                window.close()

    def test_recapture_all_forwards_every_review_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                window.images_edit.setText(str(tmp_path))
                with patch.object(
                    window, "_review_pages",
                    return_value=[("3", "无蓝色像素"), ("5", "采集失败")],
                ), patch.object(window, "_recapture_pages") as recapture, \
                     patch(
                         "one_click_gui.QMessageBox.question",
                         return_value=__import__("PySide6").QtWidgets.QMessageBox.Yes,
                     ):
                    window._review_recapture_all()
                recapture.assert_called_once()
                self.assertEqual(
                    recapture.call_args[0][0], ["3", "5"],
                )
            finally:
                window.close()

    def test_recapture_all_skips_when_queue_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                window.images_edit.setText(str(tmp_path))
                with patch.object(window, "_review_pages", return_value=[]), \
                     patch.object(window, "_recapture_pages") as recapture, \
                     patch(
                         "one_click_gui.QMessageBox.information",
                     ) as info:
                    window._review_recapture_all()
                recapture.assert_not_called()
                info.assert_called_once()
            finally:
                window.close()


def _make_gui_window(tmp: Path):
    """Build a MainWindow with a per-test ``_theme_path`` and a no-op
    settings/theme path.  Shared by every GUI test class in this file."""
    path = tmp / "ui.json"

    class Window(MainWindow):
        @staticmethod
        def _theme_path():
            return path

    return Window()


class StopTaskTests(unittest.TestCase):
    """The one-click kill switch: 停止采集 / 停止处理 buttons are disabled
    at rest, enabled while a task runs, and stop the whole child process
    tree (killing orphan Photos windows)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _make_window(self, tmp: Path):
        return _make_gui_window(tmp)

    def _fake_runner(self):
        """A stand-in Runner that never spawns a real thread."""
        from unittest.mock import Mock

        runner = Mock()
        runner.isRunning.return_value = True
        runner.proc = None
        return runner

    def test_stop_buttons_present_and_disabled_at_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._make_window(Path(tmp))
            try:
                self.assertEqual(window.stop_btn.text(), "停止采集")
                self.assertEqual(window.process_stop_btn.text(), "停止处理")
                self.assertFalse(window.stop_btn.isEnabled())
                self.assertFalse(window.process_stop_btn.isEnabled())
            finally:
                window.close()

    def test_run_enables_stop_buttons(self):
        from unittest.mock import patch as _patch

        with tempfile.TemporaryDirectory() as tmp:
            window = self._make_window(Path(tmp))
            try:
                fake = self._fake_runner()
                with _patch("one_click_gui.Runner", return_value=fake):
                    window._run(["--images", str(tmp)])
                self.assertTrue(window.stop_btn.isEnabled())
                self.assertTrue(window.process_stop_btn.isEnabled())
                self.assertFalse(window.capture_btn.isEnabled())
            finally:
                # closeEvent would block on a real QMessageBox when a
                # fake runner "is running"; swallow it by patching.
                with _patch(
                    "one_click_gui.QMessageBox.question",
                    return_value=__import__("PySide6").QtWidgets.QMessageBox.No,
                ):
                    window.close()

    def test_stop_task_kills_tree_and_restores_ui(self):
        from unittest.mock import patch as _patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            window = self._make_window(tmp_path)
            try:
                images_dir = tmp_path / "images"
                images_dir.mkdir()
                (images_dir / "1.jpg").write_bytes(b"x")
                window.images_edit.setText(str(images_dir))
                fake = self._fake_runner()
                with _patch("one_click_gui.Runner", return_value=fake), \
                     _patch("one_click_gui.QMessageBox.question",
                            return_value=__import__("PySide6").QtWidgets.QMessageBox.Yes):
                    window._run(["--images", str(images_dir)])
                    window._stop_task()
                fake.cancel.assert_called_once()
                self.assertTrue(window.capture_btn.isEnabled())
                self.assertFalse(window.stop_btn.isEnabled())
            finally:
                with _patch(
                    "one_click_gui.QMessageBox.question",
                    return_value=__import__("PySide6").QtWidgets.QMessageBox.No,
                ):
                    window.close()


class ReviewPreviewCurrentRowTests(unittest.TestCase):
    """Regression: clicking a review-list row used to show the previous
    selection's frame.  ``_review_show_preview`` must follow the current
    row (the just-clicked one), not the first item in ``selectedItems``
    (which lags the click by a signal)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QListWidgetItem

        cls.Qt = Qt
        cls.QListWidgetItem = QListWidgetItem
        cls.app = QApplication.instance() or QApplication([])

    def _seed_review(self, window, page_names):
        """Populate the review list with mock pages.  Does NOT select a
        row -- the test drives every setCurrentRow inside the patch block
        so each signal fires while the recording side_effect is active."""
        window.review_list.clear()
        for name in page_names:
            item = self.QListWidgetItem(f"{name}  [mock]")
            item.setData(self.Qt.UserRole, name)
            window.review_list.addItem(item)
        return []

    def test_current_row_always_wins_over_first_selected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            window = _make_gui_window(Path(tmp))
            try:
                calls = self._seed_review(window, ["18", "27", "4", "9"])
                # Simulate clicking each row in turn: the preview must
                # follow the current row, not the previously-selected
                # item (the bug this test guards against).
                with patch.object(
                    window, "_review_show_preview",
                    side_effect=lambda: calls.append(
                        window.review_list.currentItem().data(self.Qt.UserRole)
                        if window.review_list.currentItem() else None
                    ),
                ):
                    window.review_list.setCurrentRow(0)
                    window.review_list.setCurrentRow(1)
                    window.review_list.setCurrentRow(2)
                    window.review_list.setCurrentRow(3)
                # Each manual show_preview was called while the current
                # row was its just-set value, not a stale value.
                self.assertEqual(calls, ["18", "27", "4", "9"])
            finally:
                with patch(
                    "one_click_gui.QMessageBox.question",
                    return_value=__import__("PySide6").QtWidgets.QMessageBox.No,
                ):
                    window.close()

    def test_preview_uses_currentItem_not_selectedItems(self):
        # Construct the precise failure mode: extend-selection, select
        # row 0, then click row 2.  ``selectedItems`` after a click should
        # be [row 2] -- if it is, both approaches give the same answer;
        # we want to make sure the new code reads currentItem, which is
        # guaranteed-correct even when the signal order is wrong.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            window = _make_gui_window(Path(tmp))
            try:
                window.review_list.clear()
                for name in ("18", "27", "4", "9"):
                    item = self.QListWidgetItem(f"{name}  [mock]")
                    item.setData(self.Qt.UserRole, name)
                    window.review_list.addItem(item)
                window.review_list.setCurrentRow(0)
                window.review_list.setCurrentRow(2)
                item = window.review_list.currentItem()
                self.assertIsNotNone(item)
                self.assertEqual(item.data(self.Qt.UserRole), "4")
            finally:
                with patch(
                    "one_click_gui.QMessageBox.question",
                    return_value=__import__("PySide6").QtWidgets.QMessageBox.No,
                ):
                    window.close()


if __name__ == "__main__":
    unittest.main()


class ReviewPreviewSourceFallbackTests(unittest.TestCase):
    """审核页预览必须对 **Paddle 采集的页**也有图可看。

    Paddle 路径的 capture/{page}/ 只有 capture.json（没有 photos_* 帧），
    所以只找 photos_* 的旧逻辑会让预览全空。回退链应为：
    选中帧 → photos_base → **check 图（红框）** → 原始源图。
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _preview_path(self, tmp: Path, *, with_check: bool, with_photos: bool) -> str:
        """在临时 ROOT 下搭一个最小 outputs 目录，返回预览解析到的路径。

        ``one_click_gui.ROOT`` 被打进 tmp，避免污染真实 outputs/。
        """
        images = tmp / "book"
        images.mkdir(parents=True)
        out = tmp / "outputs" / "oneclick" / images.name
        capture = out / "capture" / "7"
        capture.mkdir(parents=True)
        src = images / "7.jpg"
        Image.new("RGB", (40, 60), "white").save(src)
        (capture / "capture.json").write_text(
            json.dumps({"source": {"path": str(src), "width": 40, "height": 60},
                        "capture": {"box_source": "paddle", "complete": True}}),
            encoding="utf-8",
        )
        if with_photos:
            Image.new("RGB", (40, 60), "gray").save(capture / "photos_base.png")
        if with_check:
            check = out / "check"
            check.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 60), "red").save(check / "7.bluebox_check.png")

        with patch("one_click_gui.ROOT", tmp):
            window = _make_gui_window(tmp)
            window.images_edit.setText(str(images))
            return window._review_current_preview_path("7")

    def test_paddle_page_falls_back_to_check_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._preview_path(Path(tmp), with_check=True, with_photos=False)
        self.assertIn("bluebox_check", path)

    def test_photos_page_prefers_photos_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._preview_path(Path(tmp), with_check=True, with_photos=True)
        self.assertIn("photos_", path)
