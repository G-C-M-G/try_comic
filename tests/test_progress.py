import json
import tempfile
import time
import unittest
from pathlib import Path

from manga_localizer.progress import ProgressFile, RunStatus, STAGES


class ProgressFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="progress_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _read(self):
        return json.loads((self.root / "run.json").read_text(encoding="utf-8"))

    def test_init_writes_run_json(self):
        progress = ProgressFile(self.root)
        status = self._read()
        self.assertEqual(status["stage"], "idle")
        self.assertEqual(status["total_stages"], len(STAGES))
        self.assertGreater(status["pid"], 0)
        self.assertTrue(status["finished"] is False)

    def test_stage_and_progress_roundtrip(self):
        progress = ProgressFile(self.root, throttle=0)
        progress.set_stage("ocr", total=10)
        progress.progress(4, item="page 2")
        status = self._read()
        self.assertEqual(status["stage"], "ocr")
        self.assertEqual(status["stage_index"], STAGES.index("ocr"))
        self.assertEqual(status["done"], 4)
        self.assertEqual(status["total"], 10)
        self.assertEqual(status["current_item"], "page 2")

    def test_log_append_and_rolling(self):
        progress = ProgressFile(self.root, throttle=0)
        for index in range(20):
            progress.log(f"line {index}")
        status = self._read()
        self.assertEqual(len(status["logs"]), 8)
        self.assertEqual(status["logs"][-1], status["logs"][-1])
        log_text = (self.root / "run.log").read_text(encoding="utf-8")
        self.assertIn("line 19", log_text)
        self.assertIn("line 0", log_text)

    def test_eta_computed(self):
        progress = ProgressFile(self.root, throttle=0)
        progress.set_stage("typeset", total=20)
        progress.status.started_at = time.time() - 10  # pretend 10s elapsed
        progress.progress(5)
        status = self._read()
        self.assertGreater(status["eta"], 0)

    def test_done_marks_finished(self):
        progress = ProgressFile(self.root, throttle=0)
        progress.done(ok=True)
        status = self._read()
        self.assertTrue(status["finished"])
        self.assertTrue(status["ok"])

    def test_fail_records_error(self):
        progress = ProgressFile(self.root, throttle=0)
        progress.fail("boom")
        status = self._read()
        self.assertTrue(status["finished"])
        self.assertFalse(status["ok"])
        self.assertIn("boom", status["error"])

    def test_unknown_stage_rejected(self):
        progress = ProgressFile(self.root)
        with self.assertRaises(ValueError):
            progress.set_stage("nope")


if __name__ == "__main__":
    unittest.main()
