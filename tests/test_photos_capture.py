import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from photos_capture import (
    OverlayBlock,
    _capture_selection_boxes,
    _close_residual_photos_windows,
    _display_bbox_to_image,
    _glyph_clicks,
    _launch_photos,
    _normalize_text,
    detect_overlay_blocks,
    map_display_bbox_affine,
)


class PhotosCaptureTests(unittest.TestCase):
    def test_normalizes_photos_column_text(self):
        self.assertEqual(_normalize_text("槍\nに\n塗布"), "槍に塗布")
        self.assertEqual(_normalize_text("やられた !!"), "やられた！！")

    def test_maps_letterboxed_viewport_to_source_pixels(self):
        self.assertEqual(
            _display_bbox_to_image([52, 39, 78, 166], (663, 942), (985, 1400)),
            [77, 58, 116, 247],
        )

    def test_detects_white_outline_overlay_islands(self):
        base = np.full((120, 160, 3), 220, dtype=np.uint8)
        active = np.full_like(base, 128)
        active[10:55, 20:48] = base[10:55, 20:48]
        active[65:110, 100:145] = base[65:110, 100:145]

        blocks = detect_overlay_blocks(base, active)

        self.assertEqual(blocks, [OverlayBlock(20, 10, 28, 45), OverlayBlock(100, 65, 45, 45)])

    def test_finds_dark_glyph_centres_on_white_background(self):
        image = np.full((80, 80, 3), 245, dtype=np.uint8)
        image[20:40, 25:40] = 20
        image[45:65, 50:65] = 20

        self.assertEqual(_glyph_clicks(image), [(32, 30), (57, 55)])

    def test_maps_bbox_with_calibrated_affine_transform(self):
        transform = {
            "source_to_display": [[0.5, 0.0, 10.0], [0.0, 0.5, 20.0]],
            "valid": True,
        }

        self.assertEqual(
            map_display_bbox_affine([20, 30, 40, 60], transform, (200, 200)),
            [20, 20, 60, 80],
        )

    @patch("photos_capture.subprocess.Popen")
    @patch("photos_capture.os.startfile")
    def test_falls_back_to_photos_executable_without_file_association(
        self, startfile: Mock, popen: Mock
    ):
        error = OSError("no file association")
        error.winerror = 1155
        startfile.side_effect = error

        _launch_photos(r"C:\manga\8.jpg")

        popen.assert_called_once_with(
            ["explorer.exe", r"C:\manga\8.jpg"], close_fds=True
        )


class WindowCloseTests(unittest.TestCase):
    """The Photos window is opened for one page at a time.  The previous
    behaviour leaked the window on every non-PhotosNoTextError path,
    leaving a ghost window on the desktop after the batch finished.  The
    finally block in ``_capture_one_page`` guarantees the window is
    closed on every outcome (success, exception, retry-exhausted)."""

    def _set_up(self, tmp, page_count: int = 1):
        from pathlib import Path

        images = Path(tmp) / "images"
        out = Path(tmp) / "out"
        images.mkdir()
        from PIL import Image

        for index in range(page_count):
            Image.new("RGB", (40, 60), (180, 180, 180)).save(
                images / f"{index + 1}.jpg"
            )
        return images, out

    @patch("photos_capture._capture_photos_fast_page")
    @patch("photos_capture._close_window_wait")
    @patch("photos_capture._is_black_page", return_value=False)
    def test_window_closed_on_success_path(
        self, _black, close_wait, fast_page
    ):
        import tempfile
        from photos_capture import _capture_one_page

        fake_window = object()
        fast_page.return_value = (
            {
                "source": {"path": "x", "width": 40, "height": 60},
                "capture": {"selection_blocks": []},
                "timing": {},
            },
            fake_window,
        )
        with tempfile.TemporaryDirectory() as tmp:
            images, out = self._set_up(tmp)
            result, _ = _capture_one_page(
                images / "1.jpg", out, timeout=1,
                resume=False, paddle_device="cpu", max_attempts=1,
                ask_fallback=False, frames_only=True, fixed_rect=None,
            )
        self.assertEqual(result["status"], "captured")
        close_wait.assert_called_once_with(fake_window)

    @patch("photos_capture._capture_photos_fast_page")
    @patch("photos_capture._close_window_wait")
    @patch("photos_capture._is_black_page", return_value=False)
    def test_window_closed_when_all_attempts_fail(
        self, _black, close_wait, fast_page
    ):
        import tempfile
        from photos_capture import _capture_one_page

        fast_page.side_effect = RuntimeError("boom")
        with tempfile.TemporaryDirectory() as tmp:
            images, out = self._set_up(tmp)
            result, _ = _capture_one_page(
                images / "1.jpg", out, timeout=1,
                resume=False, paddle_device="cpu", max_attempts=2,
                ask_fallback=False, frames_only=True, fixed_rect=None,
            )
        # All attempts failed -> status "failed"; every attempt's inner
        # except path closed the dead window, so close_wait is called once
        # per attempt with None (the inner block nulls the local handle).
        self.assertEqual(result["status"], "failed")
        self.assertEqual(close_wait.call_count, 2)
        for call in close_wait.call_args_list:
            self.assertIsNone(call.args[0])

    @patch("photos_capture._capture_photos_fast_page")
    @patch("photos_capture._close_window_wait")
    @patch("photos_capture._is_black_page", return_value=False)
    def test_window_closed_when_unexpected_exception_escapes(
        self, _black, close_wait, fast_page
    ):
        # An exception that is NOT PhotosNoTextError (e.g. an OSError from
        # ``_close_window_wait`` itself) must still land in the outer
        # finally and close the window before propagating.
        import tempfile
        from photos_capture import _capture_one_page

        fake_window = object()
        fast_page.return_value = (
            {
                "source": {"path": "x", "width": 40, "height": 60},
                "capture": {"selection_blocks": []},
                "timing": {},
            },
            fake_window,
        )
        # The inventory write would normally happen on success -- make it
        # blow up to force the exception path through the finally.
        with tempfile.TemporaryDirectory() as tmp, \
             patch("photos_capture._write_json_atomic",
                   side_effect=OSError("disk full")):
            images, out = self._set_up(tmp)
            with self.assertRaises(OSError):
                _capture_one_page(
                    images / "1.jpg", out, timeout=1,
                    resume=False, paddle_device="cpu", max_attempts=1,
                    ask_fallback=False, frames_only=True, fixed_rect=None,
                )
        close_wait.assert_called_once_with(fake_window)


class ResidualWindowSweepTests(unittest.TestCase):
    """``_close_residual_photos_windows`` must only close windows whose
    title matches a page we are about to capture.  Any other Photos
    window the user has open stays untouched."""

    @staticmethod
    def _window(title: str):
        win = Mock()
        win.window_text.return_value = title
        return win

    def test_closes_only_matching_titles(self):
        with patch("photos_capture._close_window_wait") as close_wait, \
             patch("pywinauto.Desktop") as desktop:
            windows = [
                self._window("1.jpg"),  # in target
                self._window("9.jpg"),  # in target
                self._window("family_photo.jpg"),  # user keeps
                self._window(""),  # empty title, ignore
            ]
            desktop.return_value.windows.return_value = windows
            closed = _close_residual_photos_windows({"1.jpg", "9.jpg"})
        self.assertEqual(closed, 2)
        closed_titles = [call.args[0].window_text() for call in close_wait.call_args_list]
        self.assertEqual(closed_titles, ["1.jpg", "9.jpg"])

    def test_no_target_set_returns_zero(self):
        with patch("photos_capture._close_window_wait") as close_wait:
            self.assertEqual(_close_residual_photos_windows(set()), 0)
        close_wait.assert_not_called()

    def test_desktop_failure_returns_zero(self):
        with patch("pywinauto.Desktop") as desktop:
            desktop.return_value.windows.side_effect = RuntimeError("uia dead")
            self.assertEqual(
                _close_residual_photos_windows({"1.jpg"}), 0,
            )


class LayoutOverlapSerialRetryTests(unittest.TestCase):
    """The parallel layout guard and its serial-retry queue were removed
    (reverted to f984790 behaviour: windows grab exactly where they are,
    no overlap guard).  These tests were for the removed machinery and are
    kept as a placeholder class so the file still documents what used to
    exist -- they assert nothing."""

    def test_guard_removed(self):
        self.assertTrue(True)


def _prediction(x1, y1, x2, y2, text, confidence=None):
    return {
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "text": text,
        "confidence": confidence,
    }


class MaskedOcrEngineTests(unittest.TestCase):
    """``_run_masked_ocr`` engine selection: auto falls back to Paddle when
    Windows OCR reads nothing; forced engines raise RuntimeError instead of
    silently falling back."""

    def test_auto_uses_windows_ocr_when_it_reads_text(self):
        from photos_capture import _run_masked_ocr

        predictions = [_prediction(0, 0, 10, 10, "探知")]
        with patch("photos_capture.windows_ocr_predictions", return_value=predictions), \
             patch("photos_capture._run_paddle_whole_page") as paddle:
            result, engine = _run_masked_ocr("masked.png", "auto")
        self.assertEqual(engine, "windows-ocr")
        self.assertIs(result, predictions)
        paddle.assert_not_called()

    def test_auto_falls_back_to_paddle_when_windows_reads_nothing(self):
        from photos_capture import _run_masked_ocr

        predictions = [_prediction(0, 0, 10, 10, "術式")]
        with patch("photos_capture.windows_ocr_predictions", return_value=[]), \
             patch("photos_capture._run_paddle_whole_page", return_value=predictions) as paddle:
            result, engine = _run_masked_ocr("masked.png", "auto")
        self.assertEqual(engine, "paddle")
        self.assertIs(result, predictions)
        # 设备默认 auto（worker 侧解析：有 CUDA 版 paddle 且检测到 GPU 用
        # GPU，否则 CPU）；调用方不传时按默认值透传
        paddle.assert_called_once_with("masked.png", "auto")

    def test_forced_windows_ocr_uses_windows_only(self):
        from photos_capture import _run_masked_ocr

        predictions = [_prediction(0, 0, 10, 10, "罠")]
        with patch("photos_capture.windows_ocr_available", return_value=True), \
             patch("photos_capture.windows_ocr_predictions", return_value=predictions) as win, \
             patch("photos_capture._run_paddle_whole_page") as paddle:
            result, engine = _run_masked_ocr("masked.png", "windows-ocr")
        self.assertEqual(engine, "windows-ocr")
        self.assertIs(result, predictions)
        win.assert_called_once_with("masked.png")
        paddle.assert_not_called()

    def test_forced_windows_ocr_raises_when_it_reads_nothing(self):
        from photos_capture import _run_masked_ocr

        with patch("photos_capture.windows_ocr_available", return_value=True), \
             patch("photos_capture.windows_ocr_predictions", return_value=[]), \
             patch("photos_capture._run_paddle_whole_page") as paddle:
            with self.assertRaises(RuntimeError):
                _run_masked_ocr("masked.png", "windows-ocr")
        paddle.assert_not_called()

    def test_forced_windows_ocr_unavailable_raises(self):
        from photos_capture import _run_masked_ocr

        with patch("photos_capture.windows_ocr_available", return_value=False), \
             patch("photos_capture.windows_ocr_predictions") as win:
            with self.assertRaises(RuntimeError):
                _run_masked_ocr("masked.png", "windows-ocr")
        win.assert_not_called()

    def test_forced_paddle_uses_paddle_only(self):
        from photos_capture import _run_masked_ocr

        predictions = [_prediction(0, 0, 10, 10, "結界")]
        with patch("photos_capture._run_paddle_whole_page", return_value=predictions) as paddle, \
             patch("photos_capture.windows_ocr_predictions") as win:
            result, engine = _run_masked_ocr("masked.png", "paddle", "gpu")
        self.assertEqual(engine, "paddle")
        self.assertIs(result, predictions)
        paddle.assert_called_once_with("masked.png", "gpu")
        win.assert_not_called()

    def test_forced_paddle_raises_when_it_reads_nothing(self):
        from photos_capture import _run_masked_ocr

        with patch("photos_capture._run_paddle_whole_page", return_value=[]), \
             patch("photos_capture.windows_ocr_predictions") as win:
            with self.assertRaises(RuntimeError):
                _run_masked_ocr("masked.png", "paddle")
        win.assert_not_called()

    def test_unknown_engine_raises_value_error(self):
        from photos_capture import _run_masked_ocr

        with patch("photos_capture.windows_ocr_predictions") as win:
            with self.assertRaises(ValueError):
                _run_masked_ocr("masked.png", "bogus")
        win.assert_not_called()


class RetryChainEngineTests(unittest.TestCase):
    """The crop-retry chain follows the selected OCR engine: windows-ocr
    never submits paddle crops; paddle never does Windows crop retries."""

    def _failing_columns(self):
        return [
            {
                "id": "column_000",
                "text": "",
                "paddle_text": "",
                "paddle_confidence": None,
                "text_source": "",
                "needs_review": False,
                "image_bbox": [20, 30, 60, 90],
                "direction": "vertical",
            }
        ]

    def _source_image(self, tmp: str) -> str:
        path = os.path.join(tmp, "page.png")
        Image.new("RGB", (120, 160), (240, 240, 240)).save(path)
        return path

    def test_windows_ocr_engine_skips_paddle_crops(self):
        import tempfile

        from photos_capture import _retry_columns_with_crops

        with tempfile.TemporaryDirectory() as tmp:
            source = self._source_image(tmp)
            columns = self._failing_columns()
            with patch("photos_capture.windows_ocr_predictions", return_value=[]), \
                 patch("photos_capture._run_paddle_crops") as crops:
                _retry_columns_with_crops(source, columns, "cpu", None, "windows-ocr")
        crops.assert_not_called()

    def test_paddle_engine_skips_windows_crop_retries(self):
        import tempfile

        from photos_capture import _retry_columns_with_crops

        with tempfile.TemporaryDirectory() as tmp:
            source = self._source_image(tmp)
            columns = self._failing_columns()
            with patch("photos_capture.windows_ocr_predictions") as win, \
                 patch("photos_capture._run_paddle_crops", return_value={"crops": []}) as crops:
                _retry_columns_with_crops(source, columns, "cpu", None, "paddle")
        win.assert_not_called()
        crops.assert_called_once()

    def test_auto_engine_runs_windows_then_paddle_crops(self):
        import tempfile

        from photos_capture import _retry_columns_with_crops

        with tempfile.TemporaryDirectory() as tmp:
            source = self._source_image(tmp)
            columns = self._failing_columns()
            with patch("photos_capture.windows_ocr_predictions", return_value=[]), \
                 patch("photos_capture._run_paddle_crops", return_value={"crops": []}) as crops:
                _retry_columns_with_crops(source, columns, "cpu", None, "auto")
        crops.assert_called_once()


if __name__ == "__main__":
    unittest.main()
