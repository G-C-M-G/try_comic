import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photos_capture import capture_photos_fast_batch


def _fake_image(path: Path) -> Path:
    from PIL import Image

    image = Image.new("RGB", (100, 140), (200, 200, 200))
    image.save(path)
    return path


def _ok_manifest():
    return {
        "schema_version": 2,
        "backend": "Windows Photos Scan Text",
        "source": {"path": "x", "width": 100, "height": 140},
        "capture": {
            "complete": True,
            "inventory": [],
            "timing": {"foreground_active_seconds": 5.0},
        },
        "columns": [],
    }


class CaptureRetryTests(unittest.TestCase):
    def test_retries_until_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            _fake_image(images / "1.jpg")

            calls = {"n": 0}

            def flaky(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("Photos OCR before/after frames have different sizes")
                return _ok_manifest(), None

            with patch("photos_capture._capture_photos_fast_page", side_effect=flaky):
                report = capture_photos_fast_batch(
                    images, out, pages={"1"}, max_attempts=3, ask_fallback=False
                )
            self.assertEqual(calls["n"], 3)
            self.assertEqual(report["pages"][0]["status"], "captured")
            self.assertEqual(report["pages"][0]["attempts"], 3)
            manifest = (out / "1" / "capture.json").read_text(encoding="utf-8-sig")
            self.assertIn('"attempts": 3', manifest)

    def test_first_success_exits_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            _fake_image(images / "1.jpg")

            calls = {"n": 0}

            def always_ok(*args, **kwargs):
                calls["n"] += 1
                return _ok_manifest(), None

            with patch("photos_capture._capture_photos_fast_page", side_effect=always_ok):
                report = capture_photos_fast_batch(
                    images, out, pages={"1"}, max_attempts=3, ask_fallback=False
                )
            self.assertEqual(calls["n"], 1)
            self.assertEqual(report["pages"][0]["status"], "captured")
            self.assertEqual(report["pages"][0]["attempts"], 1)

    def test_exhausts_attempts_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            out = Path(tmp) / "out"
            images.mkdir()
            _fake_image(images / "1.jpg")

            def always_fail(*args, **kwargs):
                raise RuntimeError("boom")

            with patch("photos_capture._capture_photos_fast_page", side_effect=always_fail):
                report = capture_photos_fast_batch(
                    images, out, pages={"1"}, max_attempts=3, ask_fallback=False
                )
            page = report["pages"][0]
            self.assertEqual(page["status"], "failed")
            self.assertEqual(page["attempts"], 3)
            self.assertIn("boom", page["error"])


if __name__ == "__main__":
    unittest.main()
